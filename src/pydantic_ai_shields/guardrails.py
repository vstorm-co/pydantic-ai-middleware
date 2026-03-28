"""Guardrail capabilities for pydantic-ai agents.

Ready-to-use capabilities for safety, cost control, and permission management.
Built on pydantic-ai's native capabilities API.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_shields import CostTracking, ToolGuard

    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            CostTracking(budget_usd=5.0),
            ToolGuard(blocked=["execute"], require_approval=["write_file"]),
        ],
    )
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GuardrailError(Exception):
    """Base exception for guardrail violations."""


class InputBlocked(GuardrailError):
    """Raised when user input is blocked by a guardrail."""


class OutputBlocked(GuardrailError):
    """Raised when model output is blocked by a guardrail."""


class ToolBlocked(GuardrailError):
    """Raised when a tool call is blocked by a guardrail."""

    def __init__(self, tool_name: str, reason: str = ""):
        self.tool_name = tool_name
        self.reason = reason
        msg = f"Tool '{tool_name}' blocked"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class BudgetExceededError(GuardrailError):
    """Raised when cost budget is exceeded."""

    def __init__(self, total_cost: float, budget: float):
        self.total_cost = total_cost
        self.budget = budget
        super().__init__(f"Budget exceeded: ${total_cost:.4f} > ${budget:.4f}")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CostCallback = Callable[..., Any] | None
"""Callback for cost updates: (CostInfo) -> None."""

ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool] | bool] | None
"""Callback for tool approval: (tool_name, tool_args) -> bool."""

GuardrailFunc = Callable[..., Awaitable[bool] | bool]
"""Guardrail check function: returns True if input/output is safe."""


# ---------------------------------------------------------------------------
# CostInfo
# ---------------------------------------------------------------------------


@dataclass
class CostInfo:
    """Token usage and cost information for a run.

    Attributes:
        run_cost_usd: USD cost of this run (None if model unknown).
        total_cost_usd: Cumulative USD cost across all runs (None if model unknown).
        run_request_tokens: Input tokens for this run.
        run_response_tokens: Output tokens for this run.
        total_request_tokens: Cumulative input tokens across all runs.
        total_response_tokens: Cumulative output tokens across all runs.
        run_count: Number of completed runs so far.
    """

    run_cost_usd: float | None
    total_cost_usd: float | None
    run_request_tokens: int
    run_response_tokens: int
    total_request_tokens: int
    total_response_tokens: int
    run_count: int


# ---------------------------------------------------------------------------
# CostTracking capability
# ---------------------------------------------------------------------------


@dataclass
class CostTracking(AbstractCapability[Any]):
    """Track token usage and API costs with optional budget enforcement.

    Accumulates token usage across runs, calculates USD costs using
    genai-prices, and enforces optional budget limits.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_middleware.guardrails import CostTracking

        tracking = CostTracking(budget_usd=5.0)
        agent = Agent("openai:gpt-4.1", capabilities=[tracking])

        result = await agent.run("Hello")
        print(f"Cost so far: ${tracking.total_cost:.4f}")
        ```
    """

    model_name: str | None = None
    """Model name for cost lookup (e.g. "openai:gpt-4.1"). Auto-detected if None."""

    budget_usd: float | None = None
    """Maximum allowed cumulative cost. None = unlimited."""

    on_cost_update: CostCallback = None
    """Callback invoked after each run with CostInfo."""

    # Internal state
    _total_request_tokens: int = field(default=0, init=False, repr=False)
    _total_response_tokens: int = field(default=0, init=False, repr=False)
    _total_cost_usd: float = field(default=0.0, init=False, repr=False)
    _run_count: int = field(default=0, init=False, repr=False)
    _price_per_input: float | None = field(default=None, init=False, repr=False)
    _price_per_output: float | None = field(default=None, init=False, repr=False)
    _prices_resolved: bool = field(default=False, init=False, repr=False)

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost across all runs."""
        return self._total_cost_usd

    @property
    def total_request_tokens(self) -> int:
        """Cumulative input tokens."""
        return self._total_request_tokens

    @property
    def total_response_tokens(self) -> int:
        """Cumulative output tokens."""
        return self._total_response_tokens

    @property
    def run_count(self) -> int:
        """Number of completed runs."""
        return self._run_count

    def _resolve_prices(self, model_name: str | None = None) -> None:
        """Resolve per-token pricing from genai-prices."""
        if self._prices_resolved:
            return

        name = model_name or self.model_name
        if name is None:
            self._prices_resolved = True
            return

        try:  # pragma: no cover — depends on genai-prices database
            from genai_prices import get_model_prices

            # Parse "provider:model" format
            if ":" in name:
                _, model_id = name.split(":", 1)
            else:
                model_id = name

            prices = get_model_prices(model_id)
            if prices:
                self._price_per_input = prices.get("input", 0.0)
                self._price_per_output = prices.get("output", 0.0)
        except Exception:
            pass

        self._prices_resolved = True

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        """Calculate USD cost for given token counts."""
        if self._price_per_input is None or self._price_per_output is None:
            return None
        return input_tokens * self._price_per_input + output_tokens * self._price_per_output

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Resolve prices on first run using model info from context."""
        if not self._prices_resolved:
            model_id = getattr(ctx.model, "model_id", None)
            self._resolve_prices(str(model_id) if model_id else None)

        # Check budget before run starts
        if self.budget_usd is not None and self._total_cost_usd >= self.budget_usd:
            raise BudgetExceededError(self._total_cost_usd, self.budget_usd)

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        """Track usage after run completes."""
        usage = ctx.usage
        run_input = usage.input_tokens or 0
        run_output = usage.output_tokens or 0

        self._total_request_tokens += run_input
        self._total_response_tokens += run_output
        self._run_count += 1

        run_cost = self._calculate_cost(run_input, run_output)
        if run_cost is not None:  # pragma: no cover — requires genai-prices resolution
            self._total_cost_usd += run_cost

        # Callback
        if self.on_cost_update is not None:
            info = CostInfo(
                run_cost_usd=run_cost,
                total_cost_usd=self._total_cost_usd if run_cost is not None else None,
                run_request_tokens=run_input,
                run_response_tokens=run_output,
                total_request_tokens=self._total_request_tokens,
                total_response_tokens=self._total_response_tokens,
                run_count=self._run_count,
            )
            cb_result = self.on_cost_update(info)
            if inspect.isawaitable(cb_result):
                await cb_result

        # Check budget after run
        if (  # pragma: no cover — requires genai-prices + real token usage
            self.budget_usd is not None and self._total_cost_usd >= self.budget_usd
        ):
            raise BudgetExceededError(self._total_cost_usd, self.budget_usd)

        return result


# ---------------------------------------------------------------------------
# ToolGuard capability
# ---------------------------------------------------------------------------


@dataclass
class ToolGuard(AbstractCapability[Any]):
    """Control tool access: block tools, require approval, or allow freely.

    Uses ``prepare_tools`` to hide blocked tools from the model entirely,
    and ``before_tool_execute`` to enforce approval for sensitive tools.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_middleware.guardrails import ToolGuard

        async def ask_user(tool_name, args):
            return input(f"Allow {tool_name}? (y/n) ") == "y"

        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[ToolGuard(
                blocked=["execute"],
                require_approval=["write_file", "edit_file"],
                approval_callback=ask_user,
            )],
        )
        ```
    """

    blocked: list[str] = field(default_factory=list)
    """Tool names to block entirely (hidden from model)."""

    require_approval: list[str] = field(default_factory=list)
    """Tool names that require human approval before execution."""

    approval_callback: ApprovalCallback = None
    """Async callback: (tool_name, args) -> bool. Required when require_approval is set."""

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide blocked tools from the model."""
        if not self.blocked:
            return tool_defs
        blocked_set = set(self.blocked)
        return [td for td in tool_defs if td.name not in blocked_set]

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Check approval for sensitive tools."""
        if call.tool_name not in self.require_approval:
            return args

        if self.approval_callback is None:
            raise ToolBlocked(call.tool_name, "Approval required but no callback configured")

        result = self.approval_callback(call.tool_name, args)
        if inspect.isawaitable(result):
            result = await result

        if not result:
            raise ToolBlocked(call.tool_name, "User denied")

        return args


# ---------------------------------------------------------------------------
# InputGuard capability
# ---------------------------------------------------------------------------


@dataclass
class InputGuard(AbstractCapability[Any]):
    """Block or modify user input based on a guardrail check.

    The guard function receives the user prompt and returns True if safe.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_middleware.guardrails import InputGuard

        async def check_toxicity(prompt: str) -> bool:
            # Call moderation API...
            return True  # safe

        agent = Agent("openai:gpt-4.1", capabilities=[InputGuard(guard=check_toxicity)])
        ```
    """

    guard: GuardrailFunc | None = None
    """Function that checks input safety. Returns True if safe."""

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Check input before run starts."""
        if self.guard is None:
            return

        prompt = ctx.prompt
        if prompt is None:  # pragma: no cover — prompt always set during agent.run()
            return

        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        result = self.guard(prompt_str)
        if inspect.isawaitable(result):
            result = await result

        if not result:
            raise InputBlocked(f"Input blocked by guardrail: {prompt_str[:100]}...")


# ---------------------------------------------------------------------------
# OutputGuard capability
# ---------------------------------------------------------------------------


@dataclass
class OutputGuard(AbstractCapability[Any]):
    """Block or modify model output based on a guardrail check.

    The guard function receives the model output text and returns True if safe.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_middleware.guardrails import OutputGuard

        def no_pii(output: str) -> bool:
            return "SSN" not in output and "credit card" not in output

        agent = Agent("openai:gpt-4.1", capabilities=[OutputGuard(guard=no_pii)])
        ```
    """

    guard: GuardrailFunc | None = None
    """Function that checks output safety. Returns True if safe."""

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        """Check output after run completes."""
        if self.guard is None:
            return result

        output_str = str(result.output) if hasattr(result, "output") else str(result)
        check = self.guard(output_str)
        if inspect.isawaitable(check):
            check = await check

        if not check:
            raise OutputBlocked("Output blocked by guardrail")

        return result


# ---------------------------------------------------------------------------
# AsyncGuardrail capability
# ---------------------------------------------------------------------------


@dataclass
class AsyncGuardrail(AbstractCapability[Any]):
    """Run a guardrail concurrently with the LLM call.

    Launches the guardrail check as a background task while the model generates
    a response. If the guardrail fails before the model finishes, the run is
    short-circuited to save API costs.

    Timing modes:
    - ``"concurrent"``: Guardrail runs alongside model; fail-fast on violation
    - ``"blocking"``: Guardrail completes before model starts (traditional)
    - ``"monitoring"``: Guardrail runs after model (fire-and-forget, non-blocking)

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_middleware.guardrails import AsyncGuardrail, InputGuard

        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[AsyncGuardrail(
                guard=InputGuard(guard=check_toxicity),
                timing="concurrent",
                cancel_on_failure=True,
            )],
        )
        ```
    """

    guard: AbstractCapability[Any] | None = None
    """The guardrail capability to run asynchronously."""

    timing: Literal["concurrent", "blocking", "monitoring"] = "concurrent"
    """When to run the guardrail relative to the model call."""

    cancel_on_failure: bool = True
    """Cancel/reject output if guardrail fails (concurrent mode only)."""

    timeout: float | None = None
    """Maximum time to wait for the guardrail."""

    name: str = "AsyncGuardrail"
    """Name for logging."""

    _task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)
    _error: Exception | None = field(default=None, init=False, repr=False)

    async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> Any:
        """Wrap the entire run to manage concurrent guardrail execution."""
        if self.guard is None or self.timing == "blocking":
            # Blocking mode: run guard before, then handler
            if self.guard is not None:
                await self.guard.before_run(ctx)
            return await handler()

        if self.timing == "monitoring":
            # Run handler first, then guard after (fire-and-forget)
            result = await handler()
            if self.guard is not None:  # pragma: no branch
                asyncio.create_task(
                    self._run_guard_safe(ctx),
                    name=f"{self.name}_monitor",
                )
            return result

        # Concurrent mode: launch guard + handler in parallel
        self._error = None
        self._task = asyncio.create_task(
            self._run_guard_safe(ctx),
            name=f"{self.name}_concurrent",
        )

        try:
            result = await handler()
        except Exception:  # pragma: no cover — handler error during concurrent guard
            if self._task and not self._task.done():
                self._task.cancel()
            raise

        # Wait for guard to complete
        if self._task and not self._task.done():  # pragma: no cover — race condition
            try:
                if self.timeout is not None:
                    await asyncio.wait_for(self._task, timeout=self.timeout)
                else:
                    await self._task
            except asyncio.TimeoutError:
                self._task.cancel()
                logger.warning(f"{self.name}: Guardrail timed out")
            except asyncio.CancelledError:
                pass

        if self._error is not None and self.cancel_on_failure:
            raise InputBlocked(f"Guardrail failed: {self._error}") from self._error

        return result

    async def _run_guard_safe(self, ctx: RunContext[Any]) -> None:
        """Run guardrail capturing errors."""
        try:
            if self.guard is not None:  # pragma: no branch
                await self.guard.before_run(ctx)
        except Exception as e:
            self._error = e
            logger.debug(f"{self.name}: Guardrail error: {e}")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Capabilities
    "CostTracking",
    "ToolGuard",
    "InputGuard",
    "OutputGuard",
    "AsyncGuardrail",
    # Data
    "CostInfo",
    # Exceptions
    "GuardrailError",
    "InputBlocked",
    "OutputBlocked",
    "ToolBlocked",
    "BudgetExceededError",
]

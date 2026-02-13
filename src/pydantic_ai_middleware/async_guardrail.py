"""Async guardrail middleware for concurrent guardrail and LLM execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic

from pydantic_ai.messages import ModelMessage

from .base import AgentMiddleware, DepsT
from .exceptions import GuardrailTimeout, InputBlocked
from .strategies import GuardrailTiming

if TYPE_CHECKING:
    from .context import ScopedContext
    from .permissions import ToolPermissionResult

logger = logging.getLogger(__name__)


class AsyncGuardrailMiddleware(AgentMiddleware[DepsT], Generic[DepsT]):
    """Run guardrails concurrently with LLM calls for improved latency and cost savings.

    This middleware launches guardrail checks in the background while the LLM
    generates its response. If a guardrail fails while the LLM is still working,
    the request can be short-circuited immediately to save time and API costs.

    Example:
        ```python
        from pydantic_ai_middleware import (
            MiddlewareAgent,
            AsyncGuardrailMiddleware,
            GuardrailTiming,
        )

        # Run policy check concurrently with LLM
        async_guardrail = AsyncGuardrailMiddleware(
            guardrail=PolicyViolationDetector(),
            timing=GuardrailTiming.CONCURRENT,
            cancel_on_failure=True,  # Cancel LLM if guardrail fails
        )

        agent = MiddlewareAgent(
            agent=base_agent,
            middleware=[async_guardrail],
        )

        # Guardrail starts checking while LLM generates
        # If guardrail fails, LLM is cancelled (saves API costs)
        result = await agent.run("User input")
        ```

    Performance:
        - Sequential (blocking): guardrail_time + llm_time
        - Concurrent: max(guardrail_time, llm_time)
        - With early failure: guardrail_time (LLM cancelled, no cost)

    Timing Modes:
        - BLOCKING: Traditional - guardrail completes before LLM starts
        - CONCURRENT: Guardrail runs alongside LLM, fail-fast on violation
        - ASYNC_POST: Guardrail runs after LLM (monitoring only, non-blocking)
    """

    def __init__(
        self,
        guardrail: AgentMiddleware[DepsT],
        timing: GuardrailTiming = GuardrailTiming.CONCURRENT,
        cancel_on_failure: bool = True,
        timeout: float | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize async guardrail middleware.

        Args:
            guardrail: The guardrail middleware to run asynchronously.
                This can be any AgentMiddleware instance.
            timing: When to execute the guardrail relative to the agent.
                Defaults to CONCURRENT for best latency.
            cancel_on_failure: Whether to cancel/reject the agent output if
                the guardrail fails. Only applies to CONCURRENT timing.
            timeout: Maximum time in seconds to wait for the guardrail.
                If exceeded, raises GuardrailTimeout.
            name: Optional name for debugging and logging.
        """
        self.guardrail = guardrail
        self.timing = timing
        self.cancel_on_failure = cancel_on_failure
        self.timeout = timeout
        self.name = name or f"AsyncGuardrail[{guardrail.__class__.__name__}]"

        # Internal state for tracking the concurrent guardrail task
        self._guardrail_task: asyncio.Task[Any] | None = None
        self._guardrail_prompt: str | Sequence[Any] | None = None
        self._guardrail_error: Exception | None = None
        self._ctx: ScopedContext | None = None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AsyncGuardrailMiddleware("
            f"guardrail={self.guardrail.__class__.__name__}, "
            f"timing={self.timing.value}, "
            f"cancel_on_failure={self.cancel_on_failure})"
        )

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        """Start guardrail check based on timing mode.

        For BLOCKING: Run guardrail and wait for completion.
        For CONCURRENT: Launch guardrail as background task.
        For ASYNC_POST: Do nothing (guardrail runs after agent).

        Args:
            prompt: The user prompt.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The (possibly modified) prompt.

        Raises:
            InputBlocked: If guardrail blocks the input (BLOCKING mode).
            GuardrailTimeout: If guardrail times out (BLOCKING mode).
        """
        # Reset state from previous runs
        self._guardrail_task = None
        self._guardrail_prompt = prompt
        self._guardrail_error = None
        self._ctx = ctx  # Store context for background task

        if self.timing == GuardrailTiming.BLOCKING:
            # Traditional blocking behavior - complete before agent starts
            return await self._run_guardrail_with_timeout(prompt, deps, ctx)

        elif self.timing == GuardrailTiming.CONCURRENT:
            # Launch guardrail as background task (don't wait)
            self._guardrail_task = asyncio.create_task(
                self._run_guardrail_safe(prompt, deps, ctx),
                name=f"{self.name}_before_run",
            )
            # Return immediately - guardrail runs in background
            return prompt

        elif self.timing == GuardrailTiming.ASYNC_POST:
            # Post-check mode - don't run before_run at all
            return prompt

        return prompt  # pragma: no cover

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Check guardrail result and optionally block output.

        For CONCURRENT: Wait for guardrail and check result.
        For ASYNC_POST: Launch guardrail on output (monitoring mode).
        For BLOCKING: Already completed in before_run.

        Args:
            prompt: The original user prompt.
            output: The agent output.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The (possibly modified) output.

        Raises:
            InputBlocked: If guardrail failed and cancel_on_failure is True.
        """
        if self.timing == GuardrailTiming.CONCURRENT and self._guardrail_task:
            # Wait for the concurrent guardrail to complete
            try:
                if self.timeout is not None:
                    await asyncio.wait_for(
                        self._guardrail_task,
                        timeout=self.timeout,
                    )
                else:
                    await self._guardrail_task
            except asyncio.TimeoutError as e:
                self._guardrail_task.cancel()
                raise GuardrailTimeout(self.name, self.timeout or 0) from e
            except asyncio.CancelledError:
                # Task was cancelled externally
                logger.warning(f"{self.name}: Guardrail task was cancelled")

            # Check if guardrail captured an error
            if self._guardrail_error is not None:
                if self.cancel_on_failure:
                    # Discard LLM output and raise the guardrail error
                    raise InputBlocked(
                        f"Guardrail failed: {self._guardrail_error}"
                    ) from self._guardrail_error
                else:
                    # Log but don't block
                    logger.warning(
                        f"{self.name}: Guardrail failed but output not blocked: "
                        f"{self._guardrail_error}"
                    )

        elif self.timing == GuardrailTiming.ASYNC_POST:
            # Fire-and-forget monitoring - run guardrail on output
            asyncio.create_task(
                self._run_output_guardrail_safe(prompt, output, deps, ctx),
                name=f"{self.name}_after_run_monitor",
            )

        return output

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> list[ModelMessage]:
        """Pass through to wrapped guardrail for BLOCKING mode only.

        For CONCURRENT and ASYNC_POST, we don't intercept model requests
        since the guardrail is already running or will run after.

        Args:
            messages: The messages to send to the model.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The (possibly modified) messages.
        """
        if self.timing == GuardrailTiming.BLOCKING:
            return await self.guardrail.before_model_request(messages, deps, ctx)
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any] | ToolPermissionResult:
        """Pass through to wrapped guardrail for BLOCKING mode only.

        Args:
            tool_name: The name of the tool being called.
            tool_args: The arguments to the tool.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The (possibly modified) tool arguments or a ToolPermissionResult.
        """
        if self.timing == GuardrailTiming.BLOCKING:
            return await self.guardrail.before_tool_call(tool_name, tool_args, deps, ctx)
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Pass through to wrapped guardrail for BLOCKING mode only.

        Args:
            tool_name: The name of the tool that was called.
            tool_args: The arguments that were passed to the tool.
            result: The result from the tool.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The (possibly modified) result.
        """
        if self.timing == GuardrailTiming.BLOCKING:
            return await self.guardrail.after_tool_call(tool_name, tool_args, result, deps, ctx)
        return result

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        """Pass through to wrapped guardrail.

        Args:
            tool_name: The name of the tool that failed.
            tool_args: The arguments that were passed to the tool.
            error: The exception raised by the tool.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            A different exception to raise, or None to re-raise the original.
        """
        if self.timing == GuardrailTiming.BLOCKING:
            return await self.guardrail.on_tool_error(tool_name, tool_args, error, deps, ctx)
        return None

    async def on_error(
        self,
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        """Pass through to wrapped guardrail.

        Args:
            error: The exception that occurred.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            A different exception to raise, or None to re-raise the original.
        """
        return await self.guardrail.on_error(error, deps, ctx)

    async def _run_guardrail_with_timeout(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        """Run guardrail with optional timeout (for BLOCKING mode).

        Args:
            prompt: The user prompt.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The result from the guardrail.

        Raises:
            GuardrailTimeout: If timeout is exceeded.
        """
        try:
            if self.timeout is not None:
                return await asyncio.wait_for(
                    self.guardrail.before_run(prompt, deps, ctx),
                    timeout=self.timeout,
                )
            return await self.guardrail.before_run(prompt, deps, ctx)
        except asyncio.TimeoutError as e:
            raise GuardrailTimeout(self.name, self.timeout or 0) from e

    async def _run_guardrail_safe(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        """Run guardrail and capture any errors (for CONCURRENT mode).

        This method catches exceptions and stores them for later checking
        in after_run, rather than raising immediately.

        Args:
            prompt: The user prompt.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The result from the guardrail (or original prompt on error).
        """
        try:
            return await self.guardrail.before_run(prompt, deps, ctx)
        except Exception as e:
            # Store error for checking in after_run
            self._guardrail_error = e
            logger.debug(f"{self.name}: Guardrail error captured: {e}")
            return prompt

    async def _run_output_guardrail_safe(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Run guardrail on output for monitoring (ASYNC_POST mode).

        This is fire-and-forget - errors are logged but don't affect output.

        Args:
            prompt: The original user prompt.
            output: The agent output.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing.

        Returns:
            The result from the guardrail (or original output on error).
        """
        try:
            if self.timeout is not None:
                return await asyncio.wait_for(
                    self.guardrail.after_run(prompt, output, deps, ctx),
                    timeout=self.timeout,
                )
            return await self.guardrail.after_run(prompt, output, deps, ctx)
        except asyncio.TimeoutError:
            logger.warning(f"{self.name}: Post-check guardrail timed out after {self.timeout}s")
            return output
        except Exception as e:
            # Log but don't block in monitoring mode
            logger.warning(f"{self.name}: Post-check guardrail error: {e}")
            return output

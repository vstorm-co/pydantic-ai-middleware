"""Cost tracking middleware for automatic token usage and USD cost monitoring.

This module provides middleware that tracks token usage across agent runs,
calculates USD costs via genai-prices, supports callbacks for UI updates,
and enforces budget limits.

Example:
    ```python
    from pydantic_ai_middleware import MiddlewareAgent, MiddlewareContext
    from pydantic_ai_middleware.cost_tracking import (
        CostTrackingMiddleware,
        create_cost_tracking_middleware,
    )

    # Simple usage
    cost_mw = create_cost_tracking_middleware(
        model_name="openai:gpt-4.1",
        budget_limit_usd=5.0,
        on_cost_update=lambda info: print(f"Cost: ${info.total_cost_usd:.4f}"),
    )
    agent = MiddlewareAgent(
        agent=my_agent,
        middleware=[cost_mw],
        context=MiddlewareContext(),
    )
    ```
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from genai_prices import calc_price

from .base import AgentMiddleware
from .exceptions import BudgetExceededError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .context import ScopedContext

CostCallback = Union["Callable[[CostInfo], Any]", None]
"""Callback type for cost updates. Accepts sync or async callables."""


@dataclass
class CostInfo:
    """Information about costs for a single run and cumulative totals.

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


class CostTrackingMiddleware(AgentMiddleware[Any]):
    """Middleware that tracks token usage and API costs across agent runs.

    Accumulates token usage and (optionally) calculates USD costs using
    genai-prices. Supports callbacks for real-time cost monitoring and
    budget enforcement.

    The middleware requires a ``MiddlewareContext`` on the ``MiddlewareAgent``
    to access run usage data stored in metadata.

    Attributes:
        model_name: Model identifier for cost calculation (e.g. "openai:gpt-4.1").
            Supports "provider:model" format. None disables USD cost calculation.
        budget_limit_usd: Maximum allowed cumulative cost in USD.
            None (default) means unlimited.
        on_cost_update: Callback invoked after each run with a CostInfo object.
            Supports both sync and async callables.
    """

    def __init__(
        self,
        model_name: str | None = None,
        budget_limit_usd: float | None = None,
        on_cost_update: CostCallback = None,
    ) -> None:
        """Initialize the cost tracking middleware.

        Args:
            model_name: Model identifier for cost calculation.
            budget_limit_usd: Maximum allowed cumulative cost in USD.
            on_cost_update: Callback for cost updates after each run.
        """
        self.model_name = model_name
        self.budget_limit_usd = budget_limit_usd
        self.on_cost_update = on_cost_update

        # Cumulative state
        self._total_request_tokens: int = 0
        self._total_response_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._run_count: int = 0

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost across all runs."""
        return self._total_cost_usd

    @property
    def total_request_tokens(self) -> int:
        """Cumulative input tokens across all runs."""
        return self._total_request_tokens

    @property
    def total_response_tokens(self) -> int:
        """Cumulative output tokens across all runs."""
        return self._total_response_tokens

    @property
    def run_count(self) -> int:
        """Number of completed runs."""
        return self._run_count

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self._total_request_tokens = 0
        self._total_response_tokens = 0
        self._total_cost_usd = 0.0
        self._run_count = 0

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: Any | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        """Check budget before starting a new run.

        Raises:
            BudgetExceededError: If cumulative cost exceeds budget limit.
        """
        if (
            self.budget_limit_usd is not None
            and self._total_cost_usd >= self.budget_limit_usd
        ):
            raise BudgetExceededError(self._total_cost_usd, self.budget_limit_usd)
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: Any | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Track usage and cost after a run completes."""
        # Read usage from context metadata (set by MiddlewareAgent)
        run_request_tokens = 0
        run_response_tokens = 0
        if ctx is not None:
            run_usage = ctx.metadata.get("run_usage")
            if run_usage is not None:
                run_request_tokens = getattr(run_usage, "input_tokens", 0) or 0
                run_response_tokens = getattr(run_usage, "output_tokens", 0) or 0

        # Accumulate tokens
        self._total_request_tokens += run_request_tokens
        self._total_response_tokens += run_response_tokens
        self._run_count += 1

        # Calculate cost
        run_cost: float | None = None
        if ctx is not None:
            run_usage = ctx.metadata.get("run_usage")
            if run_usage is not None:
                run_cost = self._calc_cost(run_usage)

        if run_cost is not None:
            self._total_cost_usd += run_cost

        total_cost = self._total_cost_usd if self.model_name else None

        # Build cost info
        cost_info = CostInfo(
            run_cost_usd=run_cost,
            total_cost_usd=total_cost,
            run_request_tokens=run_request_tokens,
            run_response_tokens=run_response_tokens,
            total_request_tokens=self._total_request_tokens,
            total_response_tokens=self._total_response_tokens,
            run_count=self._run_count,
        )

        # Notify callback
        await self._notify(cost_info)

        return output

    def _calc_cost(self, usage: Any) -> float | None:
        """Calculate USD cost for a usage object using genai-prices.

        Args:
            usage: A RunUsage or similar object implementing AbstractUsage.

        Returns:
            Cost in USD, or None if model_name is not set or calc fails.
        """
        if not self.model_name:
            return None

        # Parse "provider:model" format
        provider_id: str | None = None
        model_ref: str = self.model_name
        if ":" in self.model_name:
            parts = self.model_name.split(":", 1)
            provider_id = parts[0]
            model_ref = parts[1]

        try:
            result = calc_price(usage, model_ref, provider_id=provider_id)
            return float(result.total_price)
        except Exception:
            return None

    async def _notify(self, cost_info: CostInfo) -> None:
        """Call the cost update callback if set.

        Handles both sync and async callables.
        """
        if self.on_cost_update is None:
            return

        result = self.on_cost_update(cost_info)
        if inspect.isawaitable(result):
            await result


def create_cost_tracking_middleware(
    model_name: str | None = None,
    budget_limit_usd: float | None = None,
    on_cost_update: CostCallback = None,
) -> CostTrackingMiddleware:
    """Create a CostTrackingMiddleware instance.

    This is a convenience factory function.

    Args:
        model_name: Model identifier for cost calculation (e.g. "openai:gpt-4.1").
        budget_limit_usd: Maximum allowed cumulative cost in USD.
        on_cost_update: Callback for cost updates after each run.

    Returns:
        Configured CostTrackingMiddleware instance.

    Example:
        ```python
        mw = create_cost_tracking_middleware(
            model_name="anthropic:claude-sonnet-4-5-20250929",
            budget_limit_usd=10.0,
            on_cost_update=lambda info: print(f"Total: ${info.total_cost_usd:.4f}"),
        )
        ```
    """
    return CostTrackingMiddleware(
        model_name=model_name,
        budget_limit_usd=budget_limit_usd,
        on_cost_update=on_cost_update,
    )

"""Exceptions for pydantic-ai-middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .strategies import AggregationStrategy


class MiddlewareError(Exception):
    """Base exception for middleware errors."""


class InputBlocked(MiddlewareError):
    """Raised when input is blocked by middleware."""

    def __init__(self, reason: str = "Input blocked") -> None:
        self.reason = reason
        super().__init__(reason)


class ToolBlocked(MiddlewareError):
    """Raised when a tool call is blocked by middleware."""

    def __init__(self, tool_name: str, reason: str = "Tool blocked") -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' blocked: {reason}")


class OutputBlocked(MiddlewareError):
    """Raised when output is blocked by middleware."""

    def __init__(self, reason: str = "Output blocked") -> None:
        self.reason = reason
        super().__init__(reason)


class ParallelExecutionFailed(MiddlewareError):
    """Raised when parallel middleware execution fails.

    Contains information about which middleware failed and which succeeded,
    allowing for detailed error handling and debugging.
    """

    def __init__(
        self,
        errors: list[Exception],
        results: list[Any] | None = None,
        message: str = "Parallel middleware execution failed",
    ) -> None:
        """Initialize the exception.

        Args:
            errors: List of exceptions from failed middleware.
            results: Optional list of successful results (for partial failures).
            message: Human-readable error message.
        """
        self.errors = errors
        self.results = results or []
        self.failed_count = len(errors)
        self.success_count = len(self.results)
        super().__init__(f"{message}: {self.failed_count} failed, {self.success_count} succeeded")


class GuardrailTimeout(MiddlewareError):
    """Raised when an async guardrail times out.

    This can occur when a guardrail takes longer than the configured
    timeout while running concurrently with the agent.
    """

    def __init__(self, guardrail_name: str, timeout: float) -> None:
        """Initialize the exception.

        Args:
            guardrail_name: Name/identifier of the guardrail that timed out.
            timeout: The timeout value in seconds that was exceeded.
        """
        self.guardrail_name = guardrail_name
        self.timeout = timeout
        super().__init__(f"Guardrail '{guardrail_name}' timed out after {timeout:.2f}s")


class AggregationFailed(MiddlewareError):
    """Raised when parallel results cannot be aggregated.

    This occurs when the aggregation strategy cannot produce a valid
    result from the parallel middleware outputs.
    """

    def __init__(
        self,
        strategy: AggregationStrategy,
        reason: str,
        errors: list[Exception] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            strategy: The aggregation strategy that failed.
            reason: Explanation of why aggregation failed.
            errors: Optional list of underlying errors.
        """
        self.strategy = strategy
        self.reason = reason
        self.errors = errors or []
        super().__init__(f"Aggregation failed ({strategy.value}): {reason}")

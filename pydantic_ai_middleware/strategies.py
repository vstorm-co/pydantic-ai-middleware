"""Execution strategies for parallel middleware."""

from __future__ import annotations

from enum import Enum


class AggregationStrategy(Enum):
    """How to aggregate results from parallel middleware execution.

    Determines what happens when multiple middleware run concurrently
    and produce results (or failures).
    """

    ALL_MUST_PASS = "all_must_pass"
    """All middleware must succeed. If any fails, raise the first exception.

    Use for validators where all checks must pass (e.g., profanity + PII + injection).
    """

    FIRST_SUCCESS = "first_success"
    """Return result from first successful middleware. Fail only if all fail.

    Use for fallback patterns or when any positive result is sufficient.
    """

    RACE = "race"
    """Return result from first middleware to complete, even if it errors.

    Use when speed is critical and you want the fastest response.
    """

    COLLECT_ALL = "collect_all"
    """Collect all results (including errors) and return as a list.

    Use for monitoring, logging, or when you need all outputs.
    """


class GuardrailTiming(Enum):
    """When guardrails execute relative to the agent/LLM call.

    Controls the concurrency behavior between guardrails and the main agent.
    """

    BLOCKING = "blocking"
    """Traditional blocking behavior - guardrail completes before agent starts.

    Use when the guardrail must definitely pass before any LLM cost is incurred.
    """

    CONCURRENT = "concurrent"
    """Run guardrail alongside the agent. Fail-fast if guardrail fails.

    Use when you want to save latency and can cancel the LLM if guardrail fails.
    This provides the best latency when guardrails pass, and saves costs when they fail.
    """

    ASYNC_POST = "async_post"
    """Run guardrail after agent completes (monitoring mode, non-blocking).

    Use for monitoring and logging that shouldn't block the response.
    Guardrail failures are logged but don't affect the output.
    """

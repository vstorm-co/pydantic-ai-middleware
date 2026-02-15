"""Tests for CostTrackingMiddleware."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_middleware import (
    AgentMiddleware,
    BudgetExceededError,
    MiddlewareAgent,
    MiddlewareContext,
    ScopedContext,
)
from pydantic_ai_middleware.context import HookType
from pydantic_ai_middleware.cost_tracking import (
    CostInfo,
    CostTrackingMiddleware,
    create_cost_tracking_middleware,
)

# --- Helper to create a fake scoped context with metadata ---


def _make_after_run_ctx(
    request_tokens: int = 100,
    response_tokens: int = 50,
) -> ScopedContext:
    """Create a ScopedContext with run_usage in metadata."""
    parent = MiddlewareContext()
    usage = RunUsage(input_tokens=request_tokens, output_tokens=response_tokens)
    parent.set_metadata("run_usage", usage)
    return parent.for_hook(HookType.AFTER_RUN)


def _make_before_run_ctx() -> ScopedContext:
    """Create a ScopedContext for before_run hook."""
    parent = MiddlewareContext()
    return parent.for_hook(HookType.BEFORE_RUN)


# --- Tests: Initialization ---


class TestCostTrackingInit:
    """Tests for CostTrackingMiddleware initialization."""

    def test_defaults(self) -> None:
        """Default values are all None/zero."""
        mw = CostTrackingMiddleware()
        assert mw.model_name is None
        assert mw.budget_limit_usd is None
        assert mw.on_cost_update is None
        assert mw.total_cost == 0.0
        assert mw.total_request_tokens == 0
        assert mw.total_response_tokens == 0
        assert mw.run_count == 0

    def test_custom_model(self) -> None:
        """Model name is stored."""
        mw = CostTrackingMiddleware(model_name="openai:gpt-4.1")
        assert mw.model_name == "openai:gpt-4.1"

    def test_custom_budget(self) -> None:
        """Budget limit is stored."""
        mw = CostTrackingMiddleware(budget_limit_usd=5.0)
        assert mw.budget_limit_usd == 5.0

    def test_custom_callback(self) -> None:
        """Callback is stored."""
        cb = MagicMock()
        mw = CostTrackingMiddleware(on_cost_update=cb)
        assert mw.on_cost_update is cb

    def test_is_agent_middleware(self) -> None:
        """Inherits from AgentMiddleware."""
        mw = CostTrackingMiddleware()
        assert isinstance(mw, AgentMiddleware)


# --- Tests: after_run ---


class TestAfterRun:
    """Tests for after_run hook tracking."""

    async def test_accumulates_tokens(self) -> None:
        """Tokens are accumulated across runs."""
        mw = CostTrackingMiddleware()
        ctx = _make_after_run_ctx(request_tokens=100, response_tokens=50)

        await mw.after_run("prompt", "output", None, ctx)

        assert mw.total_request_tokens == 100
        assert mw.total_response_tokens == 50
        assert mw.run_count == 1

    async def test_accumulates_across_runs(self) -> None:
        """Tokens accumulate across multiple runs."""
        mw = CostTrackingMiddleware()

        ctx1 = _make_after_run_ctx(request_tokens=100, response_tokens=50)
        await mw.after_run("p1", "o1", None, ctx1)

        ctx2 = _make_after_run_ctx(request_tokens=200, response_tokens=100)
        await mw.after_run("p2", "o2", None, ctx2)

        assert mw.total_request_tokens == 300
        assert mw.total_response_tokens == 150
        assert mw.run_count == 2

    async def test_returns_output_unchanged(self) -> None:
        """Output is returned without modification."""
        mw = CostTrackingMiddleware()
        ctx = _make_after_run_ctx()

        result = await mw.after_run("prompt", "my output", None, ctx)
        assert result == "my output"

    async def test_without_context(self) -> None:
        """Works without context (tokens = 0)."""
        mw = CostTrackingMiddleware()

        await mw.after_run("prompt", "output", None, None)

        assert mw.total_request_tokens == 0
        assert mw.total_response_tokens == 0
        assert mw.run_count == 1

    async def test_without_run_usage_in_metadata(self) -> None:
        """Works when run_usage is not in metadata."""
        mw = CostTrackingMiddleware()
        parent = MiddlewareContext()
        ctx = parent.for_hook(HookType.AFTER_RUN)

        await mw.after_run("prompt", "output", None, ctx)

        assert mw.total_request_tokens == 0
        assert mw.total_response_tokens == 0
        assert mw.run_count == 1


# --- Tests: before_run budget enforcement ---


class TestBeforeRun:
    """Tests for before_run budget enforcement."""

    async def test_no_budget_passes(self) -> None:
        """Without budget, before_run passes through."""
        mw = CostTrackingMiddleware()
        result = await mw.before_run("prompt", None)
        assert result == "prompt"

    async def test_under_budget_passes(self) -> None:
        """Under budget, before_run passes through."""
        mw = CostTrackingMiddleware(budget_limit_usd=10.0)
        mw._total_cost_usd = 5.0
        result = await mw.before_run("prompt", None)
        assert result == "prompt"

    async def test_at_budget_raises(self) -> None:
        """At budget limit, raises BudgetExceededError."""
        mw = CostTrackingMiddleware(budget_limit_usd=10.0)
        mw._total_cost_usd = 10.0

        with pytest.raises(BudgetExceededError) as exc_info:
            await mw.before_run("prompt", None)

        assert exc_info.value.cost == 10.0
        assert exc_info.value.budget == 10.0

    async def test_over_budget_raises(self) -> None:
        """Over budget limit, raises BudgetExceededError."""
        mw = CostTrackingMiddleware(budget_limit_usd=5.0)
        mw._total_cost_usd = 7.5

        with pytest.raises(BudgetExceededError) as exc_info:
            await mw.before_run("prompt", None)

        assert exc_info.value.cost == 7.5
        assert exc_info.value.budget == 5.0
        assert "$7.5000" in str(exc_info.value)
        assert "$5.0000" in str(exc_info.value)

    async def test_with_context(self) -> None:
        """before_run works with context."""
        mw = CostTrackingMiddleware()
        ctx = _make_before_run_ctx()
        result = await mw.before_run("prompt", None, ctx)
        assert result == "prompt"


# --- Tests: Callback ---


class TestCallback:
    """Tests for cost update callbacks."""

    async def test_sync_callback(self) -> None:
        """Sync callback is called with CostInfo."""
        received: list[CostInfo] = []

        def on_update(info: CostInfo) -> None:
            received.append(info)

        mw = CostTrackingMiddleware(on_cost_update=on_update)
        ctx = _make_after_run_ctx(request_tokens=100, response_tokens=50)

        await mw.after_run("prompt", "output", None, ctx)

        assert len(received) == 1
        info = received[0]
        assert info.run_request_tokens == 100
        assert info.run_response_tokens == 50
        assert info.total_request_tokens == 100
        assert info.total_response_tokens == 50
        assert info.run_count == 1

    async def test_async_callback(self) -> None:
        """Async callback is awaited."""
        received: list[CostInfo] = []

        async def on_update(info: CostInfo) -> None:
            received.append(info)

        mw = CostTrackingMiddleware(on_cost_update=on_update)
        ctx = _make_after_run_ctx(request_tokens=200, response_tokens=100)

        await mw.after_run("prompt", "output", None, ctx)

        assert len(received) == 1
        assert received[0].run_request_tokens == 200

    async def test_callback_cost_fields(self) -> None:
        """CostInfo has correct cost fields when model is set."""
        received: list[CostInfo] = []

        mw = CostTrackingMiddleware(
            model_name="openai:gpt-4.1",
            on_cost_update=lambda info: received.append(info),
        )
        ctx = _make_after_run_ctx(request_tokens=1000, response_tokens=500)
        await mw.after_run("prompt", "output", None, ctx)

        assert len(received) == 1
        info = received[0]
        # With a valid model, cost should be calculated (or None if genai-prices doesn't know it)
        # We just check the types are correct
        assert info.run_cost_usd is None or isinstance(info.run_cost_usd, float)
        assert info.total_cost_usd is None or isinstance(info.total_cost_usd, float)

    async def test_no_callback(self) -> None:
        """No callback is fine."""
        mw = CostTrackingMiddleware()
        ctx = _make_after_run_ctx()
        # Should not raise
        await mw.after_run("prompt", "output", None, ctx)

    async def test_callback_without_model_has_none_costs(self) -> None:
        """Without model_name, cost fields are None."""
        received: list[CostInfo] = []

        mw = CostTrackingMiddleware(on_cost_update=lambda info: received.append(info))
        ctx = _make_after_run_ctx()
        await mw.after_run("prompt", "output", None, ctx)

        assert len(received) == 1
        assert received[0].run_cost_usd is None
        assert received[0].total_cost_usd is None


# --- Tests: Cost calculation ---


class TestCostCalc:
    """Tests for _calc_cost method."""

    def test_without_model_returns_none(self) -> None:
        """No model_name returns None."""
        mw = CostTrackingMiddleware()
        usage = RunUsage(input_tokens=100, output_tokens=50)
        assert mw._calc_cost(usage) is None

    def test_with_provider_model_format(self) -> None:
        """'provider:model' format is parsed correctly."""
        # Use a model that genai-prices should know about
        mw = CostTrackingMiddleware(model_name="openai:gpt-4o")
        usage = RunUsage(input_tokens=1000, output_tokens=500)
        cost = mw._calc_cost(usage)
        # gpt-4o should be a known model, so cost should be a float
        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0

    def test_with_plain_model_name(self) -> None:
        """Plain model name (no provider) is used directly."""
        mw = CostTrackingMiddleware(model_name="gpt-4o")
        usage = RunUsage(input_tokens=1000, output_tokens=500)
        cost = mw._calc_cost(usage)
        # Should still work since genai-prices can resolve it
        assert cost is None or isinstance(cost, float)

    def test_unknown_model_returns_none(self) -> None:
        """Unknown model returns None."""
        mw = CostTrackingMiddleware(model_name="unknown-provider:nonexistent-model-xyz")
        usage = RunUsage(input_tokens=100, output_tokens=50)
        cost = mw._calc_cost(usage)
        assert cost is None

    def test_zero_tokens(self) -> None:
        """Zero tokens returns zero or near-zero cost."""
        mw = CostTrackingMiddleware(model_name="openai:gpt-4o")
        usage = RunUsage(input_tokens=0, output_tokens=0)
        cost = mw._calc_cost(usage)
        if cost is not None:
            assert cost == 0.0


# --- Tests: Reset ---


class TestReset:
    """Tests for reset method."""

    async def test_reset_zeroes_everything(self) -> None:
        """Reset clears all accumulators."""
        mw = CostTrackingMiddleware(model_name="openai:gpt-4o")
        ctx = _make_after_run_ctx(request_tokens=100, response_tokens=50)

        await mw.after_run("prompt", "output", None, ctx)
        assert mw.run_count == 1
        assert mw.total_request_tokens == 100

        mw.reset()

        assert mw.total_cost == 0.0
        assert mw.total_request_tokens == 0
        assert mw.total_response_tokens == 0
        assert mw.run_count == 0

    def test_reset_on_fresh_instance(self) -> None:
        """Reset on fresh instance is a no-op."""
        mw = CostTrackingMiddleware()
        mw.reset()
        assert mw.total_cost == 0.0
        assert mw.run_count == 0


# --- Tests: Factory ---


class TestFactory:
    """Tests for create_cost_tracking_middleware factory."""

    def test_defaults(self) -> None:
        """Factory with no args creates default instance."""
        mw = create_cost_tracking_middleware()
        assert isinstance(mw, CostTrackingMiddleware)
        assert mw.model_name is None
        assert mw.budget_limit_usd is None
        assert mw.on_cost_update is None

    def test_with_all_args(self) -> None:
        """Factory passes all args through."""
        cb = MagicMock()
        mw = create_cost_tracking_middleware(
            model_name="anthropic:claude-sonnet-4-5-20250929",
            budget_limit_usd=10.0,
            on_cost_update=cb,
        )
        assert mw.model_name == "anthropic:claude-sonnet-4-5-20250929"
        assert mw.budget_limit_usd == 10.0
        assert mw.on_cost_update is cb

    def test_returns_correct_type(self) -> None:
        """Factory returns CostTrackingMiddleware."""
        mw = create_cost_tracking_middleware(model_name="openai:gpt-4o")
        assert isinstance(mw, CostTrackingMiddleware)
        assert isinstance(mw, AgentMiddleware)


# --- Tests: Integration with MiddlewareAgent ---


class TestIntegrationWithAgent:
    """Full integration tests with MiddlewareAgent."""

    async def test_cost_tracked_in_run(self) -> None:
        """Cost is tracked during MiddlewareAgent.run()."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        received: list[CostInfo] = []
        cost_mw = CostTrackingMiddleware(on_cost_update=lambda info: received.append(info))
        ctx = MiddlewareContext()

        middleware_agent = MiddlewareAgent(
            agent=agent,
            middleware=[cost_mw],
            context=ctx,
        )

        await middleware_agent.run("test")

        assert cost_mw.run_count == 1
        assert len(received) == 1
        # TestModel produces some usage
        assert received[0].run_count == 1

    async def test_budget_enforcement_in_run(self) -> None:
        """Budget is checked before each run."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        cost_mw = CostTrackingMiddleware(budget_limit_usd=0.001)
        ctx = MiddlewareContext()

        middleware_agent = MiddlewareAgent(
            agent=agent,
            middleware=[cost_mw],
            context=ctx,
        )

        # First run should succeed
        await middleware_agent.run("test")

        # Manually set cost over budget
        cost_mw._total_cost_usd = 1.0

        # Second run should fail
        with pytest.raises(BudgetExceededError):
            await middleware_agent.run("test2")

    async def test_multiple_runs_accumulate(self) -> None:
        """Multiple runs accumulate tokens."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        cost_mw = CostTrackingMiddleware()
        ctx = MiddlewareContext()

        middleware_agent = MiddlewareAgent(
            agent=agent,
            middleware=[cost_mw],
            context=ctx,
        )

        await middleware_agent.run("run1")
        tokens_after_1 = cost_mw.total_request_tokens + cost_mw.total_response_tokens

        await middleware_agent.run("run2")
        tokens_after_2 = cost_mw.total_request_tokens + cost_mw.total_response_tokens

        assert cost_mw.run_count == 2
        assert tokens_after_2 >= tokens_after_1

    async def test_works_without_context(self) -> None:
        """Cost tracking works without MiddlewareContext (tokens=0)."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        cost_mw = CostTrackingMiddleware()

        middleware_agent = MiddlewareAgent(
            agent=agent,
            middleware=[cost_mw],
            # No context
        )

        await middleware_agent.run("test")

        # Still counts the run
        assert cost_mw.run_count == 1
        # But no tokens (context was None so no metadata access)
        assert cost_mw.total_request_tokens == 0

    async def test_with_other_middleware(self) -> None:
        """Cost tracking works alongside other middleware."""

        class LoggingMW(AgentMiddleware[None]):
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def before_run(
                self,
                prompt: str | Sequence[Any],
                deps: None,
                ctx: ScopedContext | None = None,
            ) -> str | Sequence[Any]:
                self.calls.append("before")
                return prompt

        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMW()
        cost_mw = CostTrackingMiddleware()
        ctx = MiddlewareContext()

        middleware_agent = MiddlewareAgent(
            agent=agent,
            middleware=[logging_mw, cost_mw],
            context=ctx,
        )

        await middleware_agent.run("test")

        assert logging_mw.calls == ["before"]
        assert cost_mw.run_count == 1


# --- Tests: BudgetExceededError ---


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_fields(self) -> None:
        """Exception stores cost and budget."""
        err = BudgetExceededError(7.5, 5.0)
        assert err.cost == 7.5
        assert err.budget == 5.0

    def test_message(self) -> None:
        """Exception has descriptive message."""
        err = BudgetExceededError(7.5, 5.0)
        assert "7.5000" in str(err)
        assert "5.0000" in str(err)
        assert "Budget exceeded" in str(err)

    def test_is_middleware_error(self) -> None:
        """Inherits from MiddlewareError."""
        from pydantic_ai_middleware import MiddlewareError

        err = BudgetExceededError(1.0, 0.5)
        assert isinstance(err, MiddlewareError)

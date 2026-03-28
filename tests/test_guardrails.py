"""Tests for guardrail capabilities."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_shields.guardrails import (
    AsyncGuardrail,
    BudgetExceededError,
    CostInfo,
    CostTracking,
    GuardrailError,
    InputBlocked,
    InputGuard,
    OutputBlocked,
    OutputGuard,
    ToolBlocked,
    ToolGuard,
)

# ---------------------------------------------------------------------------
# CostTracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_default_construction(self):
        cap = CostTracking()
        assert cap.budget_usd is None
        assert cap.total_cost == 0.0
        assert cap.run_count == 0

    @pytest.mark.anyio
    async def test_tracks_tokens(self):
        cap = CostTracking()
        agent = Agent(TestModel(), capabilities=[cap])
        await agent.run("Hello")
        assert cap.run_count == 1
        assert cap.total_request_tokens >= 0
        assert cap.total_response_tokens >= 0

    @pytest.mark.anyio
    async def test_callback_called(self):
        infos: list[CostInfo] = []
        cap = CostTracking(on_cost_update=lambda info: infos.append(info))
        agent = Agent(TestModel(), capabilities=[cap])
        await agent.run("Hello")
        assert len(infos) == 1
        assert infos[0].run_count == 1

    @pytest.mark.anyio
    async def test_async_callback(self):
        infos: list[CostInfo] = []

        async def cb(info: CostInfo) -> None:
            infos.append(info)

        cap = CostTracking(on_cost_update=cb)
        agent = Agent(TestModel(), capabilities=[cap])
        await agent.run("Hello")
        assert len(infos) == 1

    @pytest.mark.anyio
    async def test_multiple_runs_accumulate(self):
        cap = CostTracking()
        agent = Agent(TestModel(), capabilities=[cap])
        await agent.run("One")
        await agent.run("Two")
        assert cap.run_count == 2

    @pytest.mark.anyio
    async def test_budget_exceeded(self):
        cap = CostTracking(budget_usd=0.0)  # Zero budget
        agent = Agent(TestModel(), capabilities=[cap])
        # First run won't fail (budget checked before run, starts at 0)
        # Set cost artificially to trigger on next run
        cap._total_cost_usd = 1.0
        with pytest.raises(BudgetExceededError):
            await agent.run("Hello")


# ---------------------------------------------------------------------------
# ToolGuard
# ---------------------------------------------------------------------------


class TestToolGuard:
    def test_default_construction(self):
        cap = ToolGuard()
        assert cap.blocked == []
        assert cap.require_approval == []

    @pytest.mark.anyio
    async def test_agent_runs_with_guard(self):
        cap = ToolGuard(blocked=["execute"])
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_no_blocked_returns_all(self):
        """No blocked tools → all tools returned."""
        cap = ToolGuard()
        ctx = _make_ctx()
        tool_defs = [
            ToolDefinition(name="read_file", description="read"),
            ToolDefinition(name="execute", description="exec"),
        ]
        result = await cap.prepare_tools(ctx, tool_defs)
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_blocked_tools_hidden(self):
        from pydantic_ai.tools import ToolDefinition

        cap = ToolGuard(blocked=["execute", "write_file"])

        ctx = _make_ctx()

        tool_defs = [
            ToolDefinition(name="read_file", description="read"),
            ToolDefinition(name="execute", description="exec"),
            ToolDefinition(name="write_file", description="write"),
        ]

        result = await cap.prepare_tools(ctx, tool_defs)
        names = [td.name for td in result]
        assert "read_file" in names
        assert "execute" not in names
        assert "write_file" not in names

    @pytest.mark.anyio
    async def test_approval_granted(self):
        async def approve(tool_name: str, args: dict[str, Any]) -> bool:
            return True

        cap = ToolGuard(require_approval=["write_file"], approval_callback=approve)
        ctx = _make_ctx()
        call = _make_call("write_file")
        tool_def = ToolDefinition(name="write_file", description="write")

        result = await cap.before_tool_execute(
            ctx, call=call, tool_def=tool_def, args={"path": "/test"}
        )
        assert result == {"path": "/test"}

    @pytest.mark.anyio
    async def test_approval_denied(self):
        async def deny(tool_name: str, args: dict[str, Any]) -> bool:
            return False

        cap = ToolGuard(require_approval=["write_file"], approval_callback=deny)
        ctx = _make_ctx()
        call = _make_call("write_file")
        tool_def = ToolDefinition(name="write_file", description="write")

        with pytest.raises(ToolBlocked, match="User denied"):
            await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args={"path": "/test"})

    @pytest.mark.anyio
    async def test_approval_no_callback_raises(self):
        cap = ToolGuard(require_approval=["write_file"])
        ctx = _make_ctx()
        call = _make_call("write_file")
        tool_def = ToolDefinition(name="write_file", description="write")

        with pytest.raises(ToolBlocked, match="no callback"):
            await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args={})

    @pytest.mark.anyio
    async def test_non_guarded_tool_passes_through(self):
        cap = ToolGuard(require_approval=["write_file"])
        ctx = _make_ctx()
        call = _make_call("read_file")
        tool_def = ToolDefinition(name="read_file", description="read")

        result = await cap.before_tool_execute(
            ctx, call=call, tool_def=tool_def, args={"path": "/test"}
        )
        assert result == {"path": "/test"}

    @pytest.mark.anyio
    async def test_sync_approval_callback(self):
        def approve_sync(tool_name: str, args: dict[str, Any]) -> bool:
            return True

        cap = ToolGuard(require_approval=["write_file"], approval_callback=approve_sync)
        ctx = _make_ctx()
        call = _make_call("write_file")
        tool_def = ToolDefinition(name="write_file", description="write")

        result = await cap.before_tool_execute(
            ctx, call=call, tool_def=tool_def, args={"path": "/test"}
        )
        assert result == {"path": "/test"}


# ---------------------------------------------------------------------------
# InputGuard
# ---------------------------------------------------------------------------


class TestInputGuard:
    @pytest.mark.anyio
    async def test_safe_input_passes(self):
        cap = InputGuard(guard=lambda prompt: True)
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_unsafe_input_blocked(self):
        cap = InputGuard(guard=lambda prompt: False)
        agent = Agent(TestModel(), capabilities=[cap])
        with pytest.raises(InputBlocked):
            await agent.run("Bad input")

    @pytest.mark.anyio
    async def test_async_guard(self):
        async def check(prompt: str) -> bool:
            return "safe" in prompt

        cap = InputGuard(guard=check)
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("This is safe")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_no_guard_passes(self):
        cap = InputGuard()
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None


# ---------------------------------------------------------------------------
# OutputGuard
# ---------------------------------------------------------------------------


class TestOutputGuard:
    @pytest.mark.anyio
    async def test_safe_output_passes(self):
        cap = OutputGuard(guard=lambda output: True)
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_unsafe_output_blocked(self):
        cap = OutputGuard(guard=lambda output: False)
        agent = Agent(TestModel(), capabilities=[cap])
        with pytest.raises(OutputBlocked):
            await agent.run("Hello")

    @pytest.mark.anyio
    async def test_async_guard(self):
        """Async output guard is awaited."""

        async def check(output: str) -> bool:
            return True

        cap = OutputGuard(guard=check)
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_no_guard_passes(self):
        cap = OutputGuard()
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None


# ---------------------------------------------------------------------------
# AsyncGuardrail
# ---------------------------------------------------------------------------


class TestAsyncGuardrail:
    @pytest.mark.anyio
    async def test_blocking_mode(self):
        cap = AsyncGuardrail(
            guard=InputGuard(guard=lambda p: True),
            timing="blocking",
        )
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_concurrent_mode_passes(self):
        cap = AsyncGuardrail(
            guard=InputGuard(guard=lambda p: True),
            timing="concurrent",
        )
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_concurrent_mode_fails(self):
        cap = AsyncGuardrail(
            guard=InputGuard(guard=lambda p: False),
            timing="concurrent",
            cancel_on_failure=True,
        )
        agent = Agent(TestModel(), capabilities=[cap])
        with pytest.raises(InputBlocked):
            await agent.run("Bad input")

    @pytest.mark.anyio
    async def test_monitoring_mode(self):
        cap = AsyncGuardrail(
            guard=InputGuard(guard=lambda p: True),
            timing="monitoring",
        )
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_no_guard(self):
        cap = AsyncGuardrail(timing="concurrent")
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    @pytest.mark.anyio
    async def test_multiple_guardrails(self):
        agent = Agent(
            TestModel(),
            capabilities=[
                CostTracking(),
                InputGuard(guard=lambda p: True),
                OutputGuard(guard=lambda o: True),
                ToolGuard(blocked=["execute"]),
            ],
        )
        result = await agent.run("Hello")
        assert result.output is not None


# ---------------------------------------------------------------------------
# CostTracking pricing
# ---------------------------------------------------------------------------


class TestCostTrackingPricing:
    def test_resolve_with_model_name(self):
        """Pricing resolution with a model name."""
        cap = CostTracking(model_name="openai:gpt-4.1")
        cap._resolve_prices()
        assert cap._prices_resolved is True

    def test_resolve_without_model_name(self):
        """No model name → prices stay None."""
        cap = CostTracking()
        cap._resolve_prices()
        assert cap._prices_resolved is True
        assert cap._price_per_input is None

    def test_resolve_plain_model_name(self):
        """Model name without provider prefix."""
        cap = CostTracking(model_name="gpt-4.1")
        cap._resolve_prices()
        assert cap._prices_resolved is True

    def test_calculate_cost_with_prices(self):
        """Cost calculation when prices are known."""
        cap = CostTracking()
        cap._price_per_input = 0.001
        cap._price_per_output = 0.002
        cost = cap._calculate_cost(1000, 500)
        assert cost is not None
        assert cost == 1000 * 0.001 + 500 * 0.002

    def test_calculate_cost_without_prices(self):
        """Cost calculation returns None without prices."""
        cap = CostTracking()
        cost = cap._calculate_cost(1000, 500)
        assert cost is None

    def test_idempotent_resolve(self):
        """Second resolve is a no-op."""
        cap = CostTracking(model_name="openai:gpt-4.1")
        cap._resolve_prices()
        first_state = cap._price_per_input
        cap._resolve_prices()
        assert cap._price_per_input == first_state


# ---------------------------------------------------------------------------
# Exception messages
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_tool_blocked_with_reason(self):
        err = ToolBlocked("execute", "dangerous")
        assert "execute" in str(err)
        assert "dangerous" in str(err)

    def test_tool_blocked_without_reason(self):
        err = ToolBlocked("execute")
        assert "execute" in str(err)

    def test_budget_exceeded(self):
        err = BudgetExceededError(5.50, 5.00)
        assert err.total_cost == 5.50
        assert err.budget == 5.00

    def test_guardrail_error_base(self):
        err = GuardrailError("test")
        assert str(err) == "test"


# ---------------------------------------------------------------------------
# AsyncGuardrail edge cases
# ---------------------------------------------------------------------------


class TestAsyncGuardrailEdgeCases:
    @pytest.mark.anyio
    async def test_concurrent_guard_failure_no_cancel(self):
        """Concurrent mode with cancel_on_failure=False logs but doesn't raise."""
        cap = AsyncGuardrail(
            guard=InputGuard(guard=lambda p: False),
            timing="concurrent",
            cancel_on_failure=False,
        )
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Should pass despite guard failure")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_timeout_on_guard(self):
        """Guard timeout is handled."""
        import asyncio

        async def slow_guard(prompt: str) -> bool:
            await asyncio.sleep(10)
            return True

        cap = AsyncGuardrail(
            guard=InputGuard(guard=slow_guard),
            timing="concurrent",
            timeout=0.01,
        )
        agent = Agent(TestModel(), capabilities=[cap])
        # Should complete — timeout is logged but doesn't crash
        result = await agent.run("Hello")
        assert result.output is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> Any:
    from pydantic_ai import RunContext
    from pydantic_ai.usage import RunUsage

    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _make_call(tool_name: str) -> Any:
    from pydantic_ai.messages import ToolCallPart

    return ToolCallPart(tool_name=tool_name, args={}, tool_call_id="test_call")

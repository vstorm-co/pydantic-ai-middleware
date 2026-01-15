"""Tests for ConditionalMiddleware."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    InputBlocked,
    MiddlewareAgent,
    MiddlewareContext,
    ScopedContext,
)
from pydantic_ai_middleware.conditional import ConditionalMiddleware
from pydantic_ai_middleware.context import HookType


class TrackingMiddleware(AgentMiddleware[None]):
    """Middleware that tracks all hook invocations."""

    def __init__(self, name: str = "tracker") -> None:
        self.name = name
        self.before_run_calls: list[str | Sequence[Any]] = []
        self.after_run_calls: list[tuple[str | Sequence[Any], Any]] = []
        self.before_model_request_calls: list[list[ModelMessage]] = []
        self.before_tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.after_tool_calls: list[tuple[str, dict[str, Any], Any]] = []
        self.on_error_calls: list[Exception] = []

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        self.before_run_calls.append(prompt)
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        self.after_run_calls.append((prompt, output))
        return output

    async def before_model_request(
        self, messages: list[ModelMessage], deps: None, ctx: ScopedContext | None = None
    ) -> list[ModelMessage]:
        self.before_model_request_calls.append(messages)
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        self.before_tool_calls.append((tool_name, tool_args))
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        self.after_tool_calls.append((tool_name, tool_args, result))
        return result

    async def on_error(
        self, error: Exception, deps: None, ctx: ScopedContext | None = None
    ) -> Exception | None:
        self.on_error_calls.append(error)
        return None


class ModifyingMiddleware(AgentMiddleware[None]):
    """Middleware that modifies values with a prefix."""

    def __init__(self, prefix: str = "modified") -> None:
        self.prefix = prefix

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        if isinstance(prompt, str):
            return f"{self.prefix}: {prompt}"
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        return f"{self.prefix}: {output}"


class BlockingMiddleware(AgentMiddleware[None]):
    """Middleware that blocks with InputBlocked."""

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        raise InputBlocked("Blocked by conditional")


class ErrorConvertingMiddleware(AgentMiddleware[None]):
    """Middleware that converts errors."""

    async def on_error(
        self, error: Exception, deps: None, ctx: ScopedContext | None = None
    ) -> Exception | None:
        return RuntimeError(f"Converted: {error}")


class TestConditionalMiddlewareInit:
    """Tests for ConditionalMiddleware initialization."""

    def test_init_with_single_middleware(self) -> None:
        """Test init with single middleware for when_true."""
        mw = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=mw,
        )
        assert cond.when_true == [mw]
        assert cond.when_false is None

    def test_init_with_sequence_middleware(self) -> None:
        """Test init with sequence of middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=[mw1, mw2],
        )
        assert cond.when_true == [mw1, mw2]

    def test_init_with_when_false(self) -> None:
        """Test init with when_false middleware."""
        mw_true = TrackingMiddleware("true")
        mw_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=mw_true,
            when_false=mw_false,
        )
        assert cond.when_true == [mw_true]
        assert cond.when_false == [mw_false]

    def test_init_with_when_false_sequence(self) -> None:
        """Test init with sequence for when_false."""
        mw_true = TrackingMiddleware("true")
        mw_false1 = TrackingMiddleware("false1")
        mw_false2 = TrackingMiddleware("false2")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=mw_true,
            when_false=[mw_false1, mw_false2],
        )
        assert cond.when_false == [mw_false1, mw_false2]

    def test_init_invalid_when_true_type(self) -> None:
        """Test init with invalid when_true type raises TypeError."""
        with pytest.raises(TypeError, match="Expected AgentMiddleware"):
            ConditionalMiddleware(
                condition=lambda ctx: True,
                when_true="not a middleware",  # type: ignore[arg-type]
            )

    def test_init_invalid_when_true_non_sequence_type(self) -> None:
        """Test init with invalid non-sequence when_true type raises TypeError."""
        with pytest.raises(TypeError, match="Expected AgentMiddleware"):
            ConditionalMiddleware(
                condition=lambda ctx: True,
                when_true=123,  # type: ignore[arg-type]
            )

    def test_init_invalid_when_false_type(self) -> None:
        """Test init with invalid when_false type raises TypeError."""
        mw = TrackingMiddleware()
        with pytest.raises(TypeError, match="Expected AgentMiddleware"):
            ConditionalMiddleware(
                condition=lambda ctx: True,
                when_true=mw,
                when_false="not a middleware",  # type: ignore[arg-type]
            )

    def test_init_invalid_sequence_item(self) -> None:
        """Test init with invalid item in sequence raises TypeError."""
        mw = TrackingMiddleware()
        with pytest.raises(TypeError, match="Expected AgentMiddleware"):
            ConditionalMiddleware(
                condition=lambda ctx: True,
                when_true=[mw, "not a middleware"],  # type: ignore[list-item]
            )


class TestConditionalMiddlewareRepr:
    """Tests for ConditionalMiddleware repr."""

    def test_repr(self) -> None:
        """Test __repr__ output."""
        mw_true = TrackingMiddleware("true")
        mw_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=mw_true,
            when_false=mw_false,
        )
        repr_str = repr(cond)
        assert "ConditionalMiddleware" in repr_str
        assert "when_true" in repr_str
        assert "when_false" in repr_str


class TestConditionalMiddlewareSelect:
    """Tests for ConditionalMiddleware._select method."""

    def test_select_returns_when_true_on_true_condition(self) -> None:
        """Test _select returns when_true when condition is True."""
        mw_true = TrackingMiddleware("true")
        mw_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=mw_true,
            when_false=mw_false,
        )
        assert cond._select(None) == [mw_true]

    def test_select_returns_when_false_on_false_condition(self) -> None:
        """Test _select returns when_false when condition is False."""
        mw_true = TrackingMiddleware("true")
        mw_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=mw_true,
            when_false=mw_false,
        )
        assert cond._select(None) == [mw_false]

    def test_select_returns_none_when_false_not_set(self) -> None:
        """Test _select returns None when condition is False and when_false is None."""
        mw_true = TrackingMiddleware("true")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=mw_true,
        )
        assert cond._select(None) is None


class TestConditionalMiddlewareBeforeRun:
    """Tests for ConditionalMiddleware.before_run hook."""

    async def test_before_run_executes_when_true(self) -> None:
        """Test before_run executes when_true middleware when condition is True."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        result = await cond.before_run("test prompt", None)

        assert result == "test prompt"
        assert tracker.before_run_calls == ["test prompt"]

    async def test_before_run_executes_when_false(self) -> None:
        """Test before_run executes when_false middleware when condition is False."""
        tracker_true = TrackingMiddleware("true")
        tracker_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker_true,
            when_false=tracker_false,
        )

        result = await cond.before_run("test prompt", None)

        assert result == "test prompt"
        assert tracker_true.before_run_calls == []
        assert tracker_false.before_run_calls == ["test prompt"]

    async def test_before_run_passthrough_when_false_none(self) -> None:
        """Test before_run passes through when condition is False and when_false is None."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        result = await cond.before_run("test prompt", None)

        assert result == "test prompt"
        assert tracker.before_run_calls == []

    async def test_before_run_executes_sequence_in_order(self) -> None:
        """Test before_run executes middleware sequence in order."""
        mw1 = ModifyingMiddleware("first")
        mw2 = ModifyingMiddleware("second")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=[mw1, mw2],
        )

        result = await cond.before_run("input", None)

        # Should apply first, then second
        assert result == "second: first: input"

    async def test_before_run_with_context(self) -> None:
        """Test before_run receives context."""
        received_ctx: list[ScopedContext | None] = []

        class ContextCapture(AgentMiddleware[None]):
            async def before_run(
                self,
                prompt: str | Sequence[Any],
                deps: None,
                ctx: ScopedContext | None = None,
            ) -> str | Sequence[Any]:
                received_ctx.append(ctx)
                return prompt

        capture = ContextCapture()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=capture,
        )

        parent_ctx = MiddlewareContext(config={"key": "value"})
        scoped = parent_ctx.for_hook(HookType.BEFORE_RUN)
        await cond.before_run("test", None, scoped)

        assert len(received_ctx) == 1
        assert received_ctx[0] is scoped


class TestConditionalMiddlewareAfterRun:
    """Tests for ConditionalMiddleware.after_run hook."""

    async def test_after_run_executes_when_true(self) -> None:
        """Test after_run executes when_true middleware when condition is True."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        result = await cond.after_run("prompt", "output", None)

        assert result == "output"
        assert tracker.after_run_calls == [("prompt", "output")]

    async def test_after_run_executes_when_false(self) -> None:
        """Test after_run executes when_false middleware when condition is False."""
        tracker_true = TrackingMiddleware("true")
        tracker_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker_true,
            when_false=tracker_false,
        )

        result = await cond.after_run("prompt", "output", None)

        assert result == "output"
        assert tracker_true.after_run_calls == []
        assert tracker_false.after_run_calls == [("prompt", "output")]

    async def test_after_run_passthrough_when_false_none(self) -> None:
        """Test after_run passes through when condition is False and when_false is None."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        result = await cond.after_run("prompt", "output", None)

        assert result == "output"
        assert tracker.after_run_calls == []

    async def test_after_run_executes_sequence_in_reverse(self) -> None:
        """Test after_run executes middleware sequence in reverse order."""
        mw1 = ModifyingMiddleware("first")
        mw2 = ModifyingMiddleware("second")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=[mw1, mw2],
        )

        result = await cond.after_run("prompt", "output", None)

        # Should apply second (reversed), then first
        assert result == "first: second: output"


class TestConditionalMiddlewareBeforeModelRequest:
    """Tests for ConditionalMiddleware.before_model_request hook."""

    async def test_before_model_request_executes_when_true(self) -> None:
        """Test before_model_request executes when_true middleware."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        result = await cond.before_model_request(messages, None)

        assert result == messages
        assert len(tracker.before_model_request_calls) == 1

    async def test_before_model_request_executes_when_false(self) -> None:
        """Test before_model_request executes when_false middleware."""
        tracker_true = TrackingMiddleware("true")
        tracker_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker_true,
            when_false=tracker_false,
        )

        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        result = await cond.before_model_request(messages, None)

        assert result == messages
        assert tracker_true.before_model_request_calls == []
        assert tracker_false.before_model_request_calls == [messages]

    async def test_before_model_request_passthrough_when_false_none(self) -> None:
        """Test before_model_request passes through when no match."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        result = await cond.before_model_request(messages, None)

        assert result == messages
        assert len(tracker.before_model_request_calls) == 0


class TestConditionalMiddlewareBeforeToolCall:
    """Tests for ConditionalMiddleware.before_tool_call hook."""

    async def test_before_tool_call_executes_when_true(self) -> None:
        """Test before_tool_call executes when_true middleware."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        result = await cond.before_tool_call("my_tool", {"arg": "value"}, None)

        assert result == {"arg": "value"}
        assert tracker.before_tool_calls == [("my_tool", {"arg": "value"})]

    async def test_before_tool_call_executes_when_false(self) -> None:
        """Test before_tool_call executes when_false middleware."""
        tracker_true = TrackingMiddleware("true")
        tracker_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker_true,
            when_false=tracker_false,
        )

        tool_args = {"arg": "value"}
        result = await cond.before_tool_call("my_tool", tool_args, None)

        assert result == tool_args
        assert tracker_true.before_tool_calls == []
        assert tracker_false.before_tool_calls == [("my_tool", tool_args)]

    async def test_before_tool_call_passthrough_when_false_none(self) -> None:
        """Test before_tool_call passes through when no match."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        result = await cond.before_tool_call("my_tool", {"arg": "value"}, None)

        assert result == {"arg": "value"}
        assert tracker.before_tool_calls == []


class TestConditionalMiddlewareAfterToolCall:
    """Tests for ConditionalMiddleware.after_tool_call hook."""

    async def test_after_tool_call_executes_when_true(self) -> None:
        """Test after_tool_call executes when_true middleware."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        result = await cond.after_tool_call("my_tool", {"arg": "value"}, "result", None)

        assert result == "result"
        assert tracker.after_tool_calls == [("my_tool", {"arg": "value"}, "result")]

    async def test_after_tool_call_executes_when_false(self) -> None:
        """Test after_tool_call executes when_false middleware."""
        tracker_true = TrackingMiddleware("true")
        tracker_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker_true,
            when_false=tracker_false,
        )

        tool_args = {"arg": "value"}
        result = await cond.after_tool_call("my_tool", tool_args, "result", None)

        assert result == "result"
        assert tracker_true.after_tool_calls == []
        assert tracker_false.after_tool_calls == [("my_tool", tool_args, "result")]

    async def test_after_tool_call_passthrough_when_false_none(self) -> None:
        """Test after_tool_call passes through when no match."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        result = await cond.after_tool_call("my_tool", {"arg": "value"}, "result", None)

        assert result == "result"
        assert tracker.after_tool_calls == []

    async def test_after_tool_call_executes_sequence_in_reverse(self) -> None:
        """Test after_tool_call executes middleware sequence in reverse order."""
        calls: list[str] = []

        class OrderTracker(AgentMiddleware[None]):
            def __init__(self, name: str) -> None:
                self.name = name

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: ScopedContext | None = None,
            ) -> Any:
                calls.append(self.name)
                return result

        mw1 = OrderTracker("first")
        mw2 = OrderTracker("second")
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=[mw1, mw2],
        )

        await cond.after_tool_call("tool", {}, "result", None)

        # Reverse order for after_* hooks
        assert calls == ["second", "first"]


class TestConditionalMiddlewareOnError:
    """Tests for ConditionalMiddleware.on_error hook."""

    async def test_on_error_executes_when_true(self) -> None:
        """Test on_error executes when_true middleware."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        error = ValueError("test error")
        result = await cond.on_error(error, None)

        assert result is None
        assert tracker.on_error_calls == [error]

    async def test_on_error_executes_when_false(self) -> None:
        """Test on_error executes when_false middleware."""
        tracker_true = TrackingMiddleware("true")
        tracker_false = TrackingMiddleware("false")
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker_true,
            when_false=tracker_false,
        )

        error = ValueError("test error")
        result = await cond.on_error(error, None)

        assert result is None
        assert tracker_true.on_error_calls == []
        assert tracker_false.on_error_calls == [error]

    async def test_on_error_passthrough_when_false_none(self) -> None:
        """Test on_error passes through when no match."""
        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        error = ValueError("test error")
        result = await cond.on_error(error, None)

        assert result is None
        assert tracker.on_error_calls == []

    async def test_on_error_can_convert_exception(self) -> None:
        """Test on_error can convert exceptions."""
        converter = ErrorConvertingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=converter,
        )

        error = ValueError("original")
        result = await cond.on_error(error, None)

        assert isinstance(result, RuntimeError)
        assert "Converted:" in str(result)


class TestConditionalMiddlewareConditionContext:
    """Tests for condition function receiving context."""

    async def test_condition_receives_context(self) -> None:
        """Test condition function receives the context."""
        received_contexts: list[ScopedContext | None] = []

        def capture_condition(ctx: ScopedContext | None) -> bool:
            received_contexts.append(ctx)
            return True

        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=capture_condition,
            when_true=tracker,
        )

        parent_ctx = MiddlewareContext(config={"test": "value"})
        scoped = parent_ctx.for_hook(HookType.BEFORE_RUN)
        await cond.before_run("test", None, scoped)

        assert len(received_contexts) == 1
        assert received_contexts[0] is scoped

    async def test_condition_based_on_config(self) -> None:
        """Test condition can use context config."""

        def use_guardrails(ctx: ScopedContext | None) -> bool:
            return bool(ctx and ctx.config.get("guardrails"))

        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=use_guardrails,
            when_true=tracker,
        )

        # With guardrails enabled
        ctx_enabled = MiddlewareContext(config={"guardrails": True})
        await cond.before_run("test", None, ctx_enabled.for_hook(HookType.BEFORE_RUN))
        assert len(tracker.before_run_calls) == 1

        # With guardrails disabled
        ctx_disabled = MiddlewareContext(config={"guardrails": False})
        await cond.before_run("test2", None, ctx_disabled.for_hook(HookType.BEFORE_RUN))
        # Should not add another call
        assert len(tracker.before_run_calls) == 1

    async def test_condition_based_on_hook_type(self) -> None:
        """Test condition can branch based on hook type."""

        def only_before_run(ctx: ScopedContext | None) -> bool:
            return ctx is not None and ctx.current_hook == HookType.BEFORE_RUN

        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=only_before_run,
            when_true=tracker,
        )

        parent_ctx = MiddlewareContext()

        # Should execute for BEFORE_RUN
        await cond.before_run("test", None, parent_ctx.for_hook(HookType.BEFORE_RUN))
        assert len(tracker.before_run_calls) == 1

        # Should NOT execute for AFTER_RUN
        await cond.after_run("test", "output", None, parent_ctx.for_hook(HookType.AFTER_RUN))
        assert len(tracker.after_run_calls) == 0

    async def test_condition_evaluated_per_hook_call(self) -> None:
        """Test condition runs for each hook invocation."""
        seen_hooks: list[HookType] = []

        def capture_hook(ctx: ScopedContext | None) -> bool:
            if ctx is not None:
                seen_hooks.append(ctx.current_hook)
            return False

        cond = ConditionalMiddleware(
            condition=capture_hook,
            when_true=TrackingMiddleware(),
        )

        parent_ctx = MiddlewareContext()
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

        await cond.before_model_request(
            messages, None, parent_ctx.for_hook(HookType.BEFORE_MODEL_REQUEST)
        )
        await cond.before_tool_call(
            "tool", {}, None, parent_ctx.for_hook(HookType.BEFORE_TOOL_CALL)
        )
        await cond.after_tool_call(
            "tool", {}, "result", None, parent_ctx.for_hook(HookType.AFTER_TOOL_CALL)
        )

        assert seen_hooks == [
            HookType.BEFORE_MODEL_REQUEST,
            HookType.BEFORE_TOOL_CALL,
            HookType.AFTER_TOOL_CALL,
        ]


class TestConditionalMiddlewareConditionErrors:
    """Tests for condition error handling."""

    async def test_condition_exception_propagates(self) -> None:
        """Test condition exceptions propagate to caller."""

        def raise_error(ctx: ScopedContext | None) -> bool:
            raise RuntimeError("condition failed")

        cond = ConditionalMiddleware(
            condition=raise_error,
            when_true=TrackingMiddleware(),
        )

        with pytest.raises(RuntimeError, match="condition failed"):
            await cond.before_run("test", None)


class TestConditionalMiddlewareIntegration:
    """Integration tests with MiddlewareAgent."""

    async def test_conditional_in_middleware_agent(self) -> None:
        """Test ConditionalMiddleware works in MiddlewareAgent."""
        model = TestModel()
        model.custom_output_text = "response"

        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=tracker,
        )

        agent = Agent(model, output_type=str)
        middleware_agent = MiddlewareAgent(agent, middleware=[cond])

        result = await middleware_agent.run("test prompt")

        assert result.output == "response"
        assert tracker.before_run_calls == ["test prompt"]

    async def test_conditional_blocking_middleware(self) -> None:
        """Test ConditionalMiddleware can block requests."""

        def block_short(ctx: ScopedContext | None) -> bool:
            return True

        blocker = BlockingMiddleware()
        cond = ConditionalMiddleware(
            condition=block_short,
            when_true=blocker,
        )

        model = TestModel()
        agent = Agent(model, output_type=str)
        middleware_agent = MiddlewareAgent(agent, middleware=[cond])

        with pytest.raises(InputBlocked, match="Blocked by conditional"):
            await middleware_agent.run("test")

    async def test_conditional_passthrough_in_agent(self) -> None:
        """Test ConditionalMiddleware passes through when condition is False."""
        model = TestModel()
        model.custom_output_text = "response"

        tracker = TrackingMiddleware()
        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=tracker,
        )

        agent = Agent(model, output_type=str)
        middleware_agent = MiddlewareAgent(agent, middleware=[cond])

        result = await middleware_agent.run("test prompt")

        assert result.output == "response"
        # Tracker should not be called
        assert tracker.before_run_calls == []
        assert tracker.after_run_calls == []

    async def test_conditional_with_context_in_agent(self) -> None:
        """Test ConditionalMiddleware with MiddlewareContext in agent."""
        model = TestModel()
        model.custom_output_text = "response"

        tracker = TrackingMiddleware()

        def check_feature_flag(ctx: ScopedContext | None) -> bool:
            return bool(ctx and ctx.config.get("feature_enabled"))

        cond = ConditionalMiddleware(
            condition=check_feature_flag,
            when_true=tracker,
        )

        agent = Agent(model, output_type=str)

        # With feature enabled
        ctx_enabled = MiddlewareContext(config={"feature_enabled": True})
        mw_agent_enabled = MiddlewareAgent(agent, middleware=[cond], context=ctx_enabled)
        await mw_agent_enabled.run("test")
        assert len(tracker.before_run_calls) == 1

        # Reset tracker
        tracker.before_run_calls.clear()

        # With feature disabled
        ctx_disabled = MiddlewareContext(config={"feature_enabled": False})
        mw_agent_disabled = MiddlewareAgent(agent, middleware=[cond], context=ctx_disabled)
        await mw_agent_disabled.run("test")
        assert len(tracker.before_run_calls) == 0

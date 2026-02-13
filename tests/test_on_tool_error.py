"""Tests for on_tool_error hook (Feature 2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from pydantic_ai_middleware import (
    AgentMiddleware,
    HookType,
    MiddlewareChain,
    MiddlewareContext,
    MiddlewareToolset,
    ScopedContext,
    on_tool_error,
)
from pydantic_ai_middleware.conditional import ConditionalMiddleware


# --- Test helpers ---


class MockToolsetTool:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name


class FailingToolset(AbstractToolset[None]):
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("tool failed")

    @property
    def id(self) -> str | None:
        return "failing"

    @property
    def label(self) -> str:
        return "FailingToolset"

    async def __aenter__(self) -> FailingToolset:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        return {}  # type: ignore

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        raise self._error

    def apply(self, visitor: Callable[[AbstractToolset[None]], None]) -> None:
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[None]], AbstractToolset[None]]
    ) -> AbstractToolset[None]:
        return visitor(self)


class SuccessToolset(AbstractToolset[None]):
    @property
    def id(self) -> str | None:
        return "success"

    @property
    def label(self) -> str:
        return "SuccessToolset"

    async def __aenter__(self) -> SuccessToolset:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        return {}  # type: ignore

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        return "ok"

    def apply(self, visitor: Callable[[AbstractToolset[None]], None]) -> None:
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[None]], AbstractToolset[None]]
    ) -> AbstractToolset[None]:
        return visitor(self)


def _mock_ctx() -> RunContext[None]:
    from unittest.mock import MagicMock

    ctx = MagicMock(spec=RunContext)
    ctx.deps = None
    return ctx


# --- Tests ---


class TestHookTypeOnToolError:
    """Tests for ON_TOOL_ERROR in HookType."""

    def test_on_tool_error_exists(self) -> None:
        assert hasattr(HookType, "ON_TOOL_ERROR")
        assert HookType.ON_TOOL_ERROR.value == 4

    def test_ordering(self) -> None:
        assert HookType.BEFORE_TOOL_CALL < HookType.ON_TOOL_ERROR
        assert HookType.ON_TOOL_ERROR < HookType.AFTER_TOOL_CALL

    def test_on_tool_error_can_read_all(self) -> None:
        """ON_TOOL_ERROR should be able to read from all hooks."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_RUN, "key", "v1")
        ctx._set_hook_data(HookType.BEFORE_TOOL_CALL, "key", "v2")

        scoped = ctx.for_hook(HookType.ON_TOOL_ERROR)
        assert scoped.get_from(HookType.BEFORE_RUN, "key") == "v1"
        assert scoped.get_from(HookType.BEFORE_TOOL_CALL, "key") == "v2"


class TestBaseOnToolError:
    """Tests for AgentMiddleware.on_tool_error default behavior."""

    async def test_default_returns_none(self) -> None:
        class MW(AgentMiddleware[None]):
            pass

        mw = MW()
        result = await mw.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None


class TestOnToolErrorInToolset:
    """Tests for on_tool_error in MiddlewareToolset."""

    async def test_on_tool_error_called_on_failure(self) -> None:
        calls: list[tuple[str, str]] = []

        class ErrorHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append((tool_name, str(error)))
                return None

        toolset = MiddlewareToolset(
            wrapped=FailingToolset(RuntimeError("boom")),
            middleware=[ErrorHandler()],
        )
        ctx = _mock_ctx()
        tool = MockToolsetTool("my_tool")

        with pytest.raises(RuntimeError, match="boom"):
            await toolset.call_tool("my_tool", {"a": 1}, ctx, tool)  # type: ignore

        assert calls == [("my_tool", "boom")]

    async def test_on_tool_error_can_replace_exception(self) -> None:
        class ErrorConverter(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                return ValueError(f"converted: {error}")

        toolset = MiddlewareToolset(
            wrapped=FailingToolset(RuntimeError("original")),
            middleware=[ErrorConverter()],
        )
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        with pytest.raises(ValueError, match="converted: original"):
            await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore

    async def test_on_tool_error_not_called_on_success(self) -> None:
        calls: list[str] = []

        class ErrorTracker(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("error_handler_called")
                return None

        toolset = MiddlewareToolset(
            wrapped=SuccessToolset(),
            middleware=[ErrorTracker()],
        )
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        result = await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore
        assert result == "ok"
        assert calls == []

    async def test_on_tool_error_respects_tool_names(self) -> None:
        calls: list[str] = []

        class EmailErrorHandler(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append(tool_name)
                return None

        toolset = MiddlewareToolset(
            wrapped=FailingToolset(),
            middleware=[EmailErrorHandler()],
        )
        ctx = _mock_ctx()
        tool = MockToolsetTool("read_file")

        # Non-matching tool - handler should NOT be called
        with pytest.raises(RuntimeError):
            await toolset.call_tool("read_file", {}, ctx, tool)  # type: ignore
        assert calls == []

        # Matching tool - handler should be called
        with pytest.raises(RuntimeError):
            await toolset.call_tool("send_email", {}, ctx, tool)  # type: ignore
        assert calls == ["send_email"]

    async def test_on_tool_error_with_context(self) -> None:
        mw_ctx = MiddlewareContext()

        class ContextHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                if ctx is not None:
                    ctx.set("error_tool", tool_name)
                    ctx.set("error_msg", str(error))
                return None

        toolset = MiddlewareToolset(
            wrapped=FailingToolset(RuntimeError("ctx_test")),
            middleware=[ContextHandler()],
            ctx=mw_ctx,
        )
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        with pytest.raises(RuntimeError):
            await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore

        scoped = mw_ctx.for_hook(HookType.ON_ERROR)
        assert scoped.get_from(HookType.ON_TOOL_ERROR, "error_tool") == "tool"
        assert scoped.get_from(HookType.ON_TOOL_ERROR, "error_msg") == "ctx_test"


class TestOnToolErrorInChain:
    """Tests for on_tool_error delegation in MiddlewareChain."""

    async def test_chain_delegates_on_tool_error(self) -> None:
        calls: list[str] = []

        class Handler1(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("h1")
                return None

        class Handler2(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("h2")
                return ValueError("handled")

        chain = MiddlewareChain([Handler1(), Handler2()])
        result = await chain.on_tool_error("tool", {}, RuntimeError("fail"), None)

        assert calls == ["h1", "h2"]
        assert isinstance(result, ValueError)

    async def test_chain_on_tool_error_all_none(self) -> None:
        class NoopHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                return None

        chain = MiddlewareChain([NoopHandler(), NoopHandler()])
        result = await chain.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None

    async def test_chain_on_tool_error_skips_non_matching(self) -> None:
        calls: list[str] = []

        class EmailHandler(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("email")
                return None

        class AllHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("all")
                return None

        chain = MiddlewareChain([EmailHandler(), AllHandler()])

        await chain.on_tool_error("read_file", {}, RuntimeError("fail"), None)
        assert calls == ["all"]

    async def test_chain_stops_at_first_handler(self) -> None:
        calls: list[str] = []

        class Handler1(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("h1")
                return ValueError("from h1")

        class Handler2(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("h2")
                return None

        chain = MiddlewareChain([Handler1(), Handler2()])
        result = await chain.on_tool_error("tool", {}, RuntimeError("fail"), None)

        assert calls == ["h1"]
        assert isinstance(result, ValueError)


class TestOnToolErrorInConditional:
    """Tests for on_tool_error delegation in ConditionalMiddleware."""

    async def test_conditional_routes_on_tool_error_with_handled(self) -> None:
        """Test that conditional returns handled exception from _run_on_tool_error."""

        class HandlerMW(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                return ValueError("handled by conditional")

        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=HandlerMW(),
        )

        result = await cond.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert isinstance(result, ValueError)
        assert "handled by conditional" in str(result)

    async def test_conditional_routes_on_tool_error(self) -> None:
        calls: list[str] = []

        class TrueHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("true")
                return None

        class FalseHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("false")
                return None

        cond = ConditionalMiddleware(
            condition=lambda ctx: True,
            when_true=TrueHandler(),
            when_false=FalseHandler(),
        )

        await cond.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert calls == ["true"]

    async def test_conditional_no_middleware_returns_none(self) -> None:
        class DummyMW(AgentMiddleware[None]):
            pass

        cond = ConditionalMiddleware(
            condition=lambda ctx: False,
            when_true=DummyMW(),
        )
        result = await cond.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None


class TestOnToolErrorInAsyncGuardrail:
    """Tests for on_tool_error delegation in AsyncGuardrailMiddleware."""

    async def test_blocking_mode_delegates(self) -> None:
        from pydantic_ai_middleware import AsyncGuardrailMiddleware
        from pydantic_ai_middleware.strategies import GuardrailTiming

        calls: list[str] = []

        class GuardrailMW(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("guardrail_handler")
                return ValueError("guardrail handled")

        guardrail = AsyncGuardrailMiddleware(
            guardrail=GuardrailMW(),
            timing=GuardrailTiming.BLOCKING,
        )

        result = await guardrail.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert calls == ["guardrail_handler"]
        assert isinstance(result, ValueError)

    async def test_non_blocking_mode_returns_none(self) -> None:
        from pydantic_ai_middleware import AsyncGuardrailMiddleware
        from pydantic_ai_middleware.strategies import GuardrailTiming

        calls: list[str] = []

        class GuardrailMW(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("should_not_be_called")
                return None

        guardrail = AsyncGuardrailMiddleware(
            guardrail=GuardrailMW(),
            timing=GuardrailTiming.CONCURRENT,
        )

        result = await guardrail.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None
        assert calls == []


class TestOnToolErrorInParallel:
    """Tests for on_tool_error delegation in ParallelMiddleware."""

    async def test_parallel_on_tool_error(self) -> None:
        from pydantic_ai_middleware import ParallelMiddleware

        calls: list[str] = []

        class Handler1(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("h1")
                return None

        class Handler2(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                calls.append("h2")
                return ValueError("from h2")

        parallel = ParallelMiddleware(middleware=[Handler1(), Handler2()])
        result = await parallel.on_tool_error("tool", {}, RuntimeError("fail"), None)

        assert "h1" in calls
        assert "h2" in calls
        assert isinstance(result, ValueError)

    async def test_parallel_on_tool_error_all_none(self) -> None:
        from pydantic_ai_middleware import ParallelMiddleware

        class NoopHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                return None

        parallel = ParallelMiddleware(middleware=[NoopHandler(), NoopHandler()])
        result = await parallel.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None

    async def test_parallel_on_tool_error_with_timeout(self) -> None:
        from pydantic_ai_middleware import ParallelMiddleware

        class Handler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                return None

        parallel = ParallelMiddleware(middleware=[Handler()], timeout=5.0)
        result = await parallel.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None

    async def test_parallel_on_tool_error_with_context(self) -> None:
        from pydantic_ai_middleware import MiddlewareContext, ParallelMiddleware

        class ContextHandler(AgentMiddleware[None]):
            async def on_tool_error(
                self, tool_name: str, tool_args: dict[str, Any], error: Exception,
                deps: None, ctx: ScopedContext | None = None
            ) -> Exception | None:
                if ctx is not None:
                    ctx.set("handled", True)
                return None

        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.ON_TOOL_ERROR)

        parallel = ParallelMiddleware(middleware=[ContextHandler()])
        await parallel.on_tool_error("tool", {}, RuntimeError("fail"), None, scoped)

        assert ctx._get_hook_data(HookType.ON_TOOL_ERROR, "handled") is True


class TestOnToolErrorDecorator:
    """Tests for @on_tool_error decorator."""

    async def test_basic_decorator(self) -> None:
        @on_tool_error
        async def handler(
            tool_name: str, tool_args: dict[str, Any], error: Exception,
            deps: None, ctx: ScopedContext | None
        ) -> Exception | None:
            return ValueError(f"handled: {tool_name}")

        assert isinstance(handler, AgentMiddleware)
        result = await handler.on_tool_error("tool", {}, RuntimeError("boom"), None)
        assert isinstance(result, ValueError)

    async def test_decorator_with_tools(self) -> None:
        @on_tool_error(tools={"send_email"})
        async def handler(
            tool_name: str, tool_args: dict[str, Any], error: Exception,
            deps: None, ctx: ScopedContext | None
        ) -> Exception | None:
            return None

        assert isinstance(handler, AgentMiddleware)
        assert handler.tool_names == {"send_email"}

    async def test_decorator_passthrough(self) -> None:
        """Other methods should be passthroughs."""

        @on_tool_error
        async def handler(
            tool_name: str, tool_args: dict[str, Any], error: Exception,
            deps: None, ctx: ScopedContext | None
        ) -> Exception | None:
            return None

        assert await handler.before_run("prompt", None) == "prompt"
        assert await handler.after_run("p", "out", None) == "out"
        assert await handler.before_tool_call("t", {}, None) == {}
        assert await handler.after_tool_call("t", {}, "r", None) == "r"
        assert await handler.on_error(ValueError(), None) is None

    async def test_default_on_tool_error_in_function_middleware(self) -> None:
        """When no on_tool_error_func is set, _FunctionMiddleware returns None."""
        from pydantic_ai_middleware.decorators import _FunctionMiddleware

        # Create a function middleware without on_tool_error_func
        mw = _FunctionMiddleware[None](before_run_func=None)
        result = await mw.on_tool_error("tool", {}, RuntimeError("fail"), None)
        assert result is None

"""Tests for hook timeout (Feature 3)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareToolset,
)
from pydantic_ai_middleware._timeout import call_with_timeout
from pydantic_ai_middleware.exceptions import MiddlewareTimeout

# --- Test helpers ---


class MockToolsetTool:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name


class MockToolset(AbstractToolset[None]):
    @property
    def id(self) -> str | None:
        return "mock"

    @property
    def label(self) -> str:
        return "MockToolset"

    async def __aenter__(self) -> MockToolset:
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


class TestCallWithTimeout:
    """Tests for the call_with_timeout utility."""

    async def test_no_timeout(self) -> None:
        async def fast() -> str:
            return "done"

        result = await call_with_timeout(fast(), None)
        assert result == "done"

    async def test_within_timeout(self) -> None:
        async def fast() -> str:
            return "done"

        result = await call_with_timeout(fast(), 5.0, "TestMW", "before_run")
        assert result == "done"

    async def test_exceeds_timeout(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(10)
            return "never"  # pragma: no cover

        with pytest.raises(MiddlewareTimeout) as exc_info:
            await call_with_timeout(slow(), 0.01, "SlowMW", "before_run")

        assert exc_info.value.middleware_name == "SlowMW"
        assert exc_info.value.hook_name == "before_run"
        assert exc_info.value.timeout == 0.01
        assert "SlowMW" in str(exc_info.value)
        assert "before_run" in str(exc_info.value)


class TestMiddlewareTimeoutException:
    """Tests for MiddlewareTimeout exception."""

    def test_attributes(self) -> None:
        exc = MiddlewareTimeout("MyMW", 5.0, "before_run")
        assert exc.middleware_name == "MyMW"
        assert exc.timeout == 5.0
        assert exc.hook_name == "before_run"

    def test_message_with_hook(self) -> None:
        exc = MiddlewareTimeout("MyMW", 1.5, "on_error")
        assert "MyMW" in str(exc)
        assert "on_error" in str(exc)
        assert "1.50s" in str(exc)

    def test_message_without_hook(self) -> None:
        exc = MiddlewareTimeout("MyMW", 2.0)
        assert "MyMW" in str(exc)
        assert "2.00s" in str(exc)


class TestTimeoutAttribute:
    """Tests for timeout attribute on AgentMiddleware."""

    def test_default_timeout_is_none(self) -> None:
        class MW(AgentMiddleware[None]):
            pass

        assert MW().timeout is None

    def test_timeout_can_be_set(self) -> None:
        class FastMW(AgentMiddleware[None]):
            timeout = 1.0

        assert FastMW().timeout == 1.0

    def test_timeout_instance_override(self) -> None:
        class MW(AgentMiddleware[None]):
            pass

        mw = MW()
        mw.timeout = 0.5
        assert mw.timeout == 0.5


class TestTimeoutInToolset:
    """Tests for timeout enforcement in MiddlewareToolset."""

    async def test_before_tool_call_timeout(self) -> None:
        class SlowMW(AgentMiddleware[None]):
            timeout = 0.01

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                await asyncio.sleep(10)
                return tool_args  # pragma: no cover

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[SlowMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        with pytest.raises(MiddlewareTimeout) as exc_info:
            await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore

        assert exc_info.value.hook_name == "before_tool_call"

    async def test_after_tool_call_timeout(self) -> None:
        class SlowAfterMW(AgentMiddleware[None]):
            timeout = 0.01

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                await asyncio.sleep(10)
                return result  # pragma: no cover

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[SlowAfterMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        with pytest.raises(MiddlewareTimeout) as exc_info:
            await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore

        assert exc_info.value.hook_name == "after_tool_call"

    async def test_on_tool_error_timeout(self) -> None:
        class SlowErrorHandler(AgentMiddleware[None]):
            timeout = 0.01

            async def on_tool_error(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                error: Exception,
                deps: None,
                ctx: Any = None,
            ) -> Exception | None:
                await asyncio.sleep(10)
                return None  # pragma: no cover

        class FailingTS(AbstractToolset[None]):
            @property
            def id(self) -> str | None:
                return "fail"

            @property
            def label(self) -> str:
                return "FailingTS"

            async def __aenter__(self) -> FailingTS:
                return self

            async def __aexit__(self, *args: Any) -> bool | None:
                return None

            async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
                return {}  # type: ignore

            async def call_tool(
                self,
                name: str,
                tool_args: dict[str, Any],
                ctx: RunContext[None],
                tool: ToolsetTool[None],
            ) -> Any:
                raise RuntimeError("fail")

            def apply(self, visitor: Callable[[AbstractToolset[None]], None]) -> None:
                visitor(self)

            def visit_and_replace(
                self, visitor: Callable[[AbstractToolset[None]], AbstractToolset[None]]
            ) -> AbstractToolset[None]:
                return visitor(self)

        toolset = MiddlewareToolset(wrapped=FailingTS(), middleware=[SlowErrorHandler()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        with pytest.raises(MiddlewareTimeout) as exc_info:
            await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore

        assert exc_info.value.hook_name == "on_tool_error"

    async def test_no_timeout_middleware_works_normally(self) -> None:
        """Middleware without timeout should work without issues."""

        class NormalMW(AgentMiddleware[None]):
            # timeout is None (default)
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                return {**tool_args, "added": True}

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[NormalMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        result = await toolset.call_tool("tool", {"x": 1}, ctx, tool)  # type: ignore
        assert result == "ok"

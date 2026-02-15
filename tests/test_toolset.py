"""Tests for MiddlewareToolset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareToolset,
    ScopedContext,
    ToolBlocked,
)


@dataclass
class MockToolsetTool:
    """Mock toolset tool for testing."""

    name: str
    description: str = "A mock tool"


class MockToolset(AbstractToolset[None]):
    """Mock toolset for testing."""

    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        self._tools = tools or {}
        self._call_results: dict[str, Any] = {}

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
        return self._tools  # type: ignore

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[None],
        tool: ToolsetTool[None],
    ) -> Any:
        return self._call_results.get(name, f"result for {name}")

    def set_call_result(self, name: str, result: Any) -> None:
        self._call_results[name] = result

    def apply(self, visitor: Callable[[AbstractToolset[None]], None]) -> None:
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[None]], AbstractToolset[None]]
    ) -> AbstractToolset[None]:
        return visitor(self)


class LoggingMiddleware(AgentMiddleware[None]):
    """Middleware that logs tool calls."""

    def __init__(self) -> None:
        self.before_calls: list[tuple[str, dict[str, Any]]] = []
        self.after_calls: list[tuple[str, Any]] = []

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        self.before_calls.append((tool_name, tool_args.copy()))
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        self.after_calls.append((tool_name, result))
        return result


class ModifyingMiddleware(AgentMiddleware[None]):
    """Middleware that modifies tool args and results."""

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        return {**tool_args, "added_before": True}

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        return {"original": result, "modified_after": True}


class BlockingMiddleware(AgentMiddleware[None]):
    """Middleware that blocks certain tools."""

    def __init__(self, blocked_tools: set[str]) -> None:
        self.blocked_tools = blocked_tools

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        if tool_name in self.blocked_tools:
            raise ToolBlocked(tool_name, "Tool is blocked")
        return tool_args


class TestMiddlewareToolset:
    """Tests for MiddlewareToolset."""

    def test_label(self) -> None:
        """Test toolset label."""
        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[])
        assert toolset.label == "MiddlewareToolset(MockToolset)"

    async def test_get_tools_delegates(self) -> None:
        """Test that get_tools delegates to wrapped toolset."""
        tools = {"tool1": MockToolsetTool(name="tool1")}
        mock = MockToolset(tools=tools)
        toolset = MiddlewareToolset(wrapped=mock, middleware=[])

        # Create a mock context
        ctx = create_mock_context()
        result = await toolset.get_tools(ctx)
        assert result == tools

    async def test_call_tool_with_logging_middleware(self) -> None:
        """Test call_tool with logging middleware."""
        mock = MockToolset()
        mock.set_call_result("my_tool", "success")

        logging_mw = LoggingMiddleware()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[logging_mw])

        ctx = create_mock_context()
        tool = MockToolsetTool(name="my_tool")
        result = await toolset.call_tool("my_tool", {"arg": "value"}, ctx, tool)  # type: ignore

        assert result == "success"
        assert logging_mw.before_calls == [("my_tool", {"arg": "value"})]
        assert logging_mw.after_calls == [("my_tool", "success")]

    async def test_call_tool_with_modifying_middleware(self) -> None:
        """Test call_tool with middleware that modifies args and results."""
        mock = MockToolset()
        mock.set_call_result("my_tool", "original_result")

        toolset = MiddlewareToolset(wrapped=mock, middleware=[ModifyingMiddleware()])

        ctx = create_mock_context()
        tool = MockToolsetTool(name="my_tool")
        result = await toolset.call_tool("my_tool", {"arg": "value"}, ctx, tool)  # type: ignore

        assert result == {"original": "original_result", "modified_after": True}

    async def test_call_tool_with_blocking_middleware(self) -> None:
        """Test call_tool with middleware that blocks tools."""
        mock = MockToolset()
        blocking_mw = BlockingMiddleware(blocked_tools={"dangerous_tool"})
        toolset = MiddlewareToolset(wrapped=mock, middleware=[blocking_mw])

        ctx = create_mock_context()
        tool = MockToolsetTool(name="dangerous_tool")

        # ToolBlocked is caught in call_tool and returned as a string
        # (so pydantic-ai treats it as a normal tool result, not an exception)
        result = await toolset.call_tool("dangerous_tool", {}, ctx, tool)  # type: ignore
        assert "blocked" in str(result).lower()
        assert "dangerous_tool" in str(result)

    async def test_call_tool_multiple_middleware_order(self) -> None:
        """Test that middleware is applied in correct order."""

        class OrderTracker(AgentMiddleware[None]):
            def __init__(self, name: str, order: list[str]) -> None:
                self.name = name
                self.order = order

            async def before_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                deps: None,
                ctx: ScopedContext | None = None,
            ) -> dict[str, Any]:
                self.order.append(f"before_{self.name}")
                return tool_args

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: ScopedContext | None = None,
            ) -> Any:
                self.order.append(f"after_{self.name}")
                return result

        order: list[str] = []
        mw1 = OrderTracker("first", order)
        mw2 = OrderTracker("second", order)
        mw3 = OrderTracker("third", order)

        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[mw1, mw2, mw3])

        ctx = create_mock_context()
        tool = MockToolsetTool(name="tool")
        await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore

        # before_* in order, after_* in reverse order
        assert order == [
            "before_first",
            "before_second",
            "before_third",
            "after_third",
            "after_second",
            "after_first",
        ]

    async def test_aenter_aexit(self) -> None:
        """Test async context manager."""
        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[])

        async with toolset as ts:
            assert ts is toolset

    def test_apply(self) -> None:
        """Test apply method."""
        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[])

        visited: list[AbstractToolset[None]] = []
        toolset.apply(lambda t: visited.append(t))

        assert mock in visited

    def test_visit_and_replace(self) -> None:
        """Test visit_and_replace method."""
        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[])

        new_mock = MockToolset()
        result = toolset.visit_and_replace(lambda t: new_mock if t is mock else t)

        assert isinstance(result, MiddlewareToolset)
        assert result.wrapped is new_mock


def create_mock_context() -> RunContext[None]:
    """Create a mock RunContext for testing."""
    # This is a simplified mock - in real tests you might need more
    from unittest.mock import MagicMock

    ctx = MagicMock(spec=RunContext)
    ctx.deps = None
    return ctx

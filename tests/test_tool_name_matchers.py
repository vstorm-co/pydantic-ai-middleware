"""Tests for tool name matchers (Feature 1)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from pydantic_ai_middleware import (
    AgentMiddleware,
    ConditionalMiddleware,
    MiddlewareChain,
    MiddlewareToolset,
    ParallelMiddleware,
    ScopedContext,
    ToolBlocked,
    ToolDecision,
    ToolPermissionResult,
    after_tool_call,
    before_tool_call,
    on_tool_error,
)


# --- Test helpers ---


class MockToolsetTool:
    name: str
    description: str = "mock"

    def __init__(self, name: str) -> None:
        self.name = name


class MockToolset(AbstractToolset[None]):
    def __init__(self) -> None:
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
        return {}  # type: ignore

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        return self._call_results.get(name, f"result:{name}")

    def set_result(self, name: str, result: Any) -> None:
        self._call_results[name] = result

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


class TestShouldHandleTool:
    """Tests for _should_handle_tool."""

    def test_none_matches_all(self) -> None:
        mw = _make_mw(tool_names=None)
        assert mw._should_handle_tool("any_tool")
        assert mw._should_handle_tool("another")

    def test_set_matches_specific(self) -> None:
        mw = _make_mw(tool_names={"send_email", "delete_file"})
        assert mw._should_handle_tool("send_email")
        assert mw._should_handle_tool("delete_file")
        assert not mw._should_handle_tool("read_file")

    def test_empty_set_matches_nothing(self) -> None:
        mw = _make_mw(tool_names=set())
        assert not mw._should_handle_tool("any_tool")


class TestToolNameFilteringInToolset:
    """Tests for tool name filtering in MiddlewareToolset."""

    async def test_middleware_skipped_for_non_matching_tool(self) -> None:
        """Middleware with tool_names should be skipped for non-matching tools."""
        calls: list[str] = []

        class TrackerMW(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append(f"before:{tool_name}")
                return tool_args

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                calls.append(f"after:{tool_name}")
                return result

        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[TrackerMW()])

        ctx = _mock_ctx()
        tool = MockToolsetTool("read_file")

        # Call with non-matching tool
        result = await toolset.call_tool("read_file", {}, ctx, tool)  # type: ignore
        assert result == "result:read_file"
        assert calls == []  # Middleware was skipped

        # Call with matching tool
        result = await toolset.call_tool("send_email", {}, ctx, tool)  # type: ignore
        assert calls == ["before:send_email", "after:send_email"]

    async def test_mixed_middleware_filtering(self) -> None:
        """Some middleware match, some don't."""
        calls: list[str] = []

        class AllToolsMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("all")
                return tool_args

        class EmailOnlyMW(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("email_only")
                return tool_args

        mock = MockToolset()
        toolset = MiddlewareToolset(wrapped=mock, middleware=[AllToolsMW(), EmailOnlyMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("read_file")

        await toolset.call_tool("read_file", {}, ctx, tool)  # type: ignore
        assert calls == ["all"]

        calls.clear()
        await toolset.call_tool("send_email", {}, ctx, tool)  # type: ignore
        assert calls == ["all", "email_only"]


class TestToolNameFilteringInChain:
    """Tests for tool name filtering in MiddlewareChain."""

    async def test_chain_filters_by_tool_name(self) -> None:
        calls: list[str] = []

        class A(AgentMiddleware[None]):
            tool_names = {"tool_a"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("A")
                return tool_args

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                calls.append("after_A")
                return result

        class B(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("B")
                return tool_args

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                calls.append("after_B")
                return result

        chain = MiddlewareChain([A(), B()])

        await chain.before_tool_call("tool_b", {}, None)
        assert calls == ["B"]

        calls.clear()
        await chain.before_tool_call("tool_a", {}, None)
        assert calls == ["A", "B"]

        calls.clear()
        await chain.after_tool_call("tool_b", {}, "result", None)
        assert calls == ["after_B"]


class TestDecoratorToolsParam:
    """Tests for tools parameter on decorators."""

    async def test_before_tool_call_with_tools(self) -> None:
        @before_tool_call(tools={"send_email"})
        async def validate_email(
            tool_name: str, tool_args: dict[str, Any], deps: None, ctx: ScopedContext | None
        ) -> dict[str, Any]:
            return {**tool_args, "validated": True}

        assert isinstance(validate_email, AgentMiddleware)
        assert validate_email.tool_names == {"send_email"}
        assert validate_email._should_handle_tool("send_email")
        assert not validate_email._should_handle_tool("other")

    async def test_after_tool_call_with_tools(self) -> None:
        @after_tool_call(tools={"read_file"})
        async def log_read(
            tool_name: str,
            tool_args: dict[str, Any],
            result: Any,
            deps: None,
            ctx: ScopedContext | None,
        ) -> Any:
            return result

        assert isinstance(log_read, AgentMiddleware)
        assert log_read.tool_names == {"read_file"}

    async def test_on_tool_error_with_tools(self) -> None:
        @on_tool_error(tools={"dangerous"})
        async def handle_danger(
            tool_name: str,
            tool_args: dict[str, Any],
            error: Exception,
            deps: None,
            ctx: ScopedContext | None,
        ) -> Exception | None:
            return RuntimeError("safe error")

        assert isinstance(handle_danger, AgentMiddleware)
        assert handle_danger.tool_names == {"dangerous"}

    async def test_before_tool_call_without_tools_plain_decorator(self) -> None:
        """Plain @before_tool_call should still work."""

        @before_tool_call
        async def validate_all(
            tool_name: str, tool_args: dict[str, Any], deps: None, ctx: ScopedContext | None
        ) -> dict[str, Any]:
            return tool_args

        assert isinstance(validate_all, AgentMiddleware)
        assert validate_all.tool_names is None


class TestChainPermissionResultPassthrough:
    """Test that chain passes through ToolPermissionResult."""

    async def test_chain_before_tool_call_non_dict_return(self) -> None:
        """When middleware in chain returns ToolPermissionResult, chain passes it through."""
        from pydantic_ai_middleware import ToolDecision, ToolPermissionResult

        class PermMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(
                    decision=ToolDecision.ALLOW,
                    modified_args={"perm": True},
                )

        chain = MiddlewareChain([PermMW()])
        result = await chain.before_tool_call("tool", {}, None)
        # Chain passes through the result (may be ToolPermissionResult)
        assert isinstance(result, ToolPermissionResult)


class TestToolNameFilteringInParallel:
    """Tests for tool name filtering in ParallelMiddleware."""

    async def test_parallel_before_tool_call_filters_by_tool_name(self) -> None:
        """ParallelMiddleware should skip middleware that don't match tool_names."""
        calls: list[str] = []

        class EmailMW(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("email")
                return tool_args

        class AllMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("all")
                return tool_args

        parallel = ParallelMiddleware(middleware=[EmailMW(), AllMW()])

        # Non-matching tool: only AllMW should execute
        await parallel.before_tool_call("read_file", {}, None)
        assert calls == ["all"]

        calls.clear()
        # Matching tool: both should execute
        await parallel.before_tool_call("send_email", {}, None)
        assert sorted(calls) == ["all", "email"]

    async def test_parallel_before_tool_call_no_matching_returns_tool_args(self) -> None:
        """When no middleware match tool_names, return original tool_args."""

        class EmailMW(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                return {**tool_args, "modified": True}

        parallel = ParallelMiddleware(middleware=[EmailMW()])
        result = await parallel.before_tool_call("other_tool", {"key": "val"}, None)
        assert result == {"key": "val"}

    async def test_parallel_after_tool_call_filters_by_tool_name(self) -> None:
        """ParallelMiddleware.after_tool_call should filter by tool_names."""
        calls: list[str] = []

        class FileMW(AgentMiddleware[None]):
            tool_names = {"write_file"}

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                calls.append("file")
                return result

        class AllMW(AgentMiddleware[None]):
            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                calls.append("all")
                return result

        parallel = ParallelMiddleware(middleware=[FileMW(), AllMW()])

        await parallel.after_tool_call("read_file", {}, "ok", None)
        assert calls == ["all"]

        calls.clear()
        await parallel.after_tool_call("write_file", {}, "ok", None)
        assert sorted(calls) == ["all", "file"]

    async def test_parallel_after_tool_call_no_matching_returns_result(self) -> None:
        """When no middleware match, return original result."""

        class FileMW(AgentMiddleware[None]):
            tool_names = {"write_file"}

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                return "modified"

        parallel = ParallelMiddleware(middleware=[FileMW()])
        result = await parallel.after_tool_call("other", {}, "original", None)
        assert result == "original"

    async def test_parallel_on_tool_error_filters_by_tool_name(self) -> None:
        """ParallelMiddleware.on_tool_error should filter by tool_names."""
        calls: list[str] = []

        class DBHandler(AgentMiddleware[None]):
            tool_names = {"db_query"}

            async def on_tool_error(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                error: Exception,
                deps: None,
                ctx: Any = None,
            ) -> Exception | None:
                calls.append("db")
                return ValueError("handled by db")

        parallel = ParallelMiddleware(middleware=[DBHandler()])

        # Non-matching: handler should NOT run
        result = await parallel.on_tool_error("read_file", {}, RuntimeError("fail"), None)
        assert result is None
        assert calls == []

        # Matching: handler should run
        result = await parallel.on_tool_error("db_query", {}, RuntimeError("fail"), None)
        assert isinstance(result, ValueError)
        assert calls == ["db"]

    async def test_parallel_before_tool_call_permission_result_deny(self) -> None:
        """ParallelMiddleware should propagate ToolPermissionResult DENY."""

        class DenyMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(decision=ToolDecision.DENY, reason="blocked")

        class AllowMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                return tool_args

        parallel = ParallelMiddleware(middleware=[DenyMW(), AllowMW()])
        result = await parallel.before_tool_call("tool", {}, None)
        # DENY should take priority over dict result
        assert isinstance(result, ToolPermissionResult)
        assert result.decision == ToolDecision.DENY

    async def test_parallel_before_tool_call_permission_result_ask(self) -> None:
        """ASK should take priority over ALLOW."""

        class AskMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(decision=ToolDecision.ASK, reason="confirm?")

        class AllowMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(decision=ToolDecision.ALLOW)

        parallel = ParallelMiddleware(middleware=[AskMW(), AllowMW()])
        result = await parallel.before_tool_call("tool", {}, None)
        assert isinstance(result, ToolPermissionResult)
        assert result.decision == ToolDecision.ASK

    async def test_parallel_before_tool_call_deny_beats_ask(self) -> None:
        """DENY should take priority over ASK."""

        class AskMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(decision=ToolDecision.ASK, reason="confirm?")

        class DenyMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(decision=ToolDecision.DENY, reason="nope")

        parallel = ParallelMiddleware(middleware=[AskMW(), DenyMW()])
        result = await parallel.before_tool_call("tool", {}, None)
        assert isinstance(result, ToolPermissionResult)
        assert result.decision == ToolDecision.DENY

    async def test_parallel_no_permission_results_aggregates_normally(self) -> None:
        """When no ToolPermissionResult, aggregate dict results normally."""

        class ModifyMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                return {**tool_args, "modified": True}

        parallel = ParallelMiddleware(middleware=[ModifyMW()])
        result = await parallel.before_tool_call("tool", {"key": "val"}, None)
        assert isinstance(result, dict)
        assert result == {"key": "val", "modified": True}


class TestToolNameFilteringInConditional:
    """Tests for tool name filtering in ConditionalMiddleware."""

    async def test_conditional_before_tool_call_filters_by_tool_name(self) -> None:
        """ConditionalMiddleware should respect tool_names on nested middleware."""
        calls: list[str] = []

        class EmailMW(AgentMiddleware[None]):
            tool_names = {"send_email"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("email")
                return tool_args

        cond = ConditionalMiddleware(condition=lambda ctx: True, when_true=EmailMW())

        await cond.before_tool_call("read_file", {}, None)
        assert calls == []

        await cond.before_tool_call("send_email", {}, None)
        assert calls == ["email"]

    async def test_conditional_after_tool_call_filters_by_tool_name(self) -> None:
        """ConditionalMiddleware.after_tool_call should respect tool_names."""
        calls: list[str] = []

        class FileMW(AgentMiddleware[None]):
            tool_names = {"write_file"}

            async def after_tool_call(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                result: Any,
                deps: None,
                ctx: Any = None,
            ) -> Any:
                calls.append("file")
                return result

        cond = ConditionalMiddleware(condition=lambda ctx: True, when_true=FileMW())

        await cond.after_tool_call("read_file", {}, "result", None)
        assert calls == []

        await cond.after_tool_call("write_file", {}, "result", None)
        assert calls == ["file"]

    async def test_conditional_on_tool_error_filters_by_tool_name(self) -> None:
        """ConditionalMiddleware.on_tool_error should respect tool_names."""

        class DBHandler(AgentMiddleware[None]):
            tool_names = {"db_query"}

            async def on_tool_error(
                self,
                tool_name: str,
                tool_args: dict[str, Any],
                error: Exception,
                deps: None,
                ctx: Any = None,
            ) -> Exception | None:
                return ValueError("handled")

        cond = ConditionalMiddleware(condition=lambda ctx: True, when_true=DBHandler())

        # Non-matching: should not handle
        result = await cond.on_tool_error("read_file", {}, RuntimeError("err"), None)
        assert result is None

        # Matching: should handle
        result = await cond.on_tool_error("db_query", {}, RuntimeError("err"), None)
        assert isinstance(result, ValueError)

    async def test_conditional_pipeline_filters_by_tool_name(self) -> None:
        """Tool name filtering in conditional with multiple middleware."""
        calls: list[str] = []

        class A(AgentMiddleware[None]):
            tool_names = {"tool_a"}

            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("A")
                return tool_args

        class B(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                calls.append("B")
                return tool_args

        cond = ConditionalMiddleware(condition=lambda ctx: True, when_true=[A(), B()])

        await cond.before_tool_call("tool_x", {}, None)
        assert calls == ["B"]  # A skipped, B matches all

        calls.clear()
        await cond.before_tool_call("tool_a", {}, None)
        assert calls == ["A", "B"]  # Both match


def _make_mw(tool_names: set[str] | None = None) -> AgentMiddleware[None]:
    class MW(AgentMiddleware[None]):
        pass

    mw = MW()
    mw.tool_names = tool_names
    return mw

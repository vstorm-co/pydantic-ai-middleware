"""Tests for permission decision protocol (Feature 4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareToolset,
    ScopedContext,
    ToolBlocked,
    ToolDecision,
    ToolPermissionResult,
)

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


class TestToolDecision:
    """Tests for ToolDecision enum."""

    def test_values(self) -> None:
        assert ToolDecision.ALLOW.value == "allow"
        assert ToolDecision.DENY.value == "deny"
        assert ToolDecision.ASK.value == "ask"


class TestToolPermissionResult:
    """Tests for ToolPermissionResult dataclass."""

    def test_allow(self) -> None:
        r = ToolPermissionResult(decision=ToolDecision.ALLOW)
        assert r.decision == ToolDecision.ALLOW
        assert r.reason == ""
        assert r.modified_args is None

    def test_deny_with_reason(self) -> None:
        r = ToolPermissionResult(decision=ToolDecision.DENY, reason="Not authorized")
        assert r.decision == ToolDecision.DENY
        assert r.reason == "Not authorized"

    def test_allow_with_modified_args(self) -> None:
        r = ToolPermissionResult(
            decision=ToolDecision.ALLOW,
            modified_args={"sanitized": True},
        )
        assert r.modified_args == {"sanitized": True}

    def test_ask_with_reason(self) -> None:
        r = ToolPermissionResult(decision=ToolDecision.ASK, reason="Needs approval")
        assert r.decision == ToolDecision.ASK
        assert r.reason == "Needs approval"


class TestPermissionResultProcessing:
    """Tests for _process_permission_result in MiddlewareToolset."""

    async def test_dict_return_passthrough(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        result = await ts._process_permission_result({"a": 1}, "tool", {"a": 0})
        assert result == {"a": 1}

    async def test_allow_without_modified_args(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        perm = ToolPermissionResult(decision=ToolDecision.ALLOW)
        result = await ts._process_permission_result(perm, "tool", {"original": True})
        assert result == {"original": True}

    async def test_allow_with_modified_args(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        perm = ToolPermissionResult(
            decision=ToolDecision.ALLOW,
            modified_args={"modified": True},
        )
        result = await ts._process_permission_result(perm, "tool", {"original": True})
        assert result == {"modified": True}

    async def test_deny_raises_tool_blocked(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        perm = ToolPermissionResult(decision=ToolDecision.DENY, reason="Not allowed")
        with pytest.raises(ToolBlocked) as exc_info:
            await ts._process_permission_result(perm, "my_tool", {})
        assert exc_info.value.tool_name == "my_tool"
        assert "Not allowed" in str(exc_info.value)

    async def test_deny_default_reason(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        perm = ToolPermissionResult(decision=ToolDecision.DENY)
        with pytest.raises(ToolBlocked, match="Permission denied"):
            await ts._process_permission_result(perm, "tool", {})

    async def test_ask_without_handler_raises_tool_blocked(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        perm = ToolPermissionResult(decision=ToolDecision.ASK, reason="Need approval")
        with pytest.raises(ToolBlocked, match="Need approval"):
            await ts._process_permission_result(perm, "tool", {})

    async def test_ask_without_handler_no_reason(self) -> None:
        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[])
        perm = ToolPermissionResult(decision=ToolDecision.ASK)
        with pytest.raises(ToolBlocked, match="no handler configured"):
            await ts._process_permission_result(perm, "tool", {})

    async def test_ask_with_handler_allowing(self) -> None:
        async def handler(tool_name: str, tool_args: dict[str, Any], reason: str) -> bool:
            return True

        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[], permission_handler=handler)
        perm = ToolPermissionResult(decision=ToolDecision.ASK, reason="Approve?")
        result = await ts._process_permission_result(perm, "tool", {"a": 1})
        assert result == {"a": 1}

    async def test_ask_with_handler_denying(self) -> None:
        async def handler(tool_name: str, tool_args: dict[str, Any], reason: str) -> bool:
            return False

        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[], permission_handler=handler)
        perm = ToolPermissionResult(decision=ToolDecision.ASK, reason="Approve?")
        with pytest.raises(ToolBlocked, match="Approve"):
            await ts._process_permission_result(perm, "tool", {})

    async def test_ask_with_handler_denying_no_reason(self) -> None:
        async def handler(tool_name: str, tool_args: dict[str, Any], reason: str) -> bool:
            return False

        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[], permission_handler=handler)
        perm = ToolPermissionResult(decision=ToolDecision.ASK)
        with pytest.raises(ToolBlocked, match="Permission denied by handler"):
            await ts._process_permission_result(perm, "tool", {})

    async def test_ask_with_handler_and_modified_args(self) -> None:
        async def handler(tool_name: str, tool_args: dict[str, Any], reason: str) -> bool:
            return True

        ts = MiddlewareToolset(wrapped=MockToolset(), middleware=[], permission_handler=handler)
        perm = ToolPermissionResult(
            decision=ToolDecision.ASK,
            reason="Approve?",
            modified_args={"sanitized": True},
        )
        result = await ts._process_permission_result(perm, "tool", {"original": True})
        assert result == {"sanitized": True}


class TestPermissionInToolCallFlow:
    """Tests for permission decisions in the full call_tool flow."""

    async def test_middleware_returns_permission_allow(self) -> None:
        class PermMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(
                    decision=ToolDecision.ALLOW,
                    modified_args={**tool_args, "approved": True},
                )

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[PermMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        result = await toolset.call_tool("tool", {"x": 1}, ctx, tool)  # type: ignore
        assert result == "ok"

    async def test_middleware_returns_permission_deny(self) -> None:
        class DenyMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                if tool_name == "dangerous":
                    return ToolPermissionResult(
                        decision=ToolDecision.DENY,
                        reason="Tool is dangerous",
                    )
                return tool_args

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[DenyMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("dangerous")

        with pytest.raises(ToolBlocked, match="Tool is dangerous"):
            await toolset.call_tool("dangerous", {}, ctx, tool)  # type: ignore

    async def test_middleware_returns_permission_ask_with_handler(self) -> None:
        handler_calls: list[tuple[str, dict[str, Any], str]] = []

        async def permission_handler(
            tool_name: str, tool_args: dict[str, Any], reason: str
        ) -> bool:
            handler_calls.append((tool_name, tool_args, reason))
            return True

        class AskMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(
                    decision=ToolDecision.ASK,
                    reason="Needs explicit approval",
                )

        toolset = MiddlewareToolset(
            wrapped=MockToolset(),
            middleware=[AskMW()],
            permission_handler=permission_handler,
        )
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        result = await toolset.call_tool("tool", {"a": 1}, ctx, tool)  # type: ignore
        assert result == "ok"
        assert handler_calls == [("tool", {"a": 1}, "Needs explicit approval")]

    async def test_backwards_compatible_dict_return(self) -> None:
        """Existing middleware that returns dict should still work."""

        class OldStyleMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                return {**tool_args, "added": True}

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[OldStyleMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        result = await toolset.call_tool("tool", {"x": 1}, ctx, tool)  # type: ignore
        assert result == "ok"

    async def test_multiple_middleware_with_permission(self) -> None:
        """Multiple middleware can use different return types."""

        class DictMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> dict[str, Any]:
                return {**tool_args, "from_dict": True}

        class PermMW(AgentMiddleware[None]):
            async def before_tool_call(
                self, tool_name: str, tool_args: dict[str, Any], deps: None, ctx: Any = None
            ) -> Any:
                return ToolPermissionResult(
                    decision=ToolDecision.ALLOW,
                    modified_args={**tool_args, "from_perm": True},
                )

        toolset = MiddlewareToolset(wrapped=MockToolset(), middleware=[DictMW(), PermMW()])
        ctx = _mock_ctx()
        tool = MockToolsetTool("tool")

        result = await toolset.call_tool("tool", {}, ctx, tool)  # type: ignore
        assert result == "ok"

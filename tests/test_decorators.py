"""Tests for decorator-based middleware."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from pydantic_ai_middleware import (
    AgentMiddleware,
    InputBlocked,
    ScopedContext,
    ToolBlocked,
    after_run,
    after_tool_call,
    before_model_request,
    before_run,
    before_tool_call,
    on_error,
)


class TestBeforeRunDecorator:
    """Tests for before_run decorator."""

    async def test_before_run_basic(self) -> None:
        """Test basic before_run decorator."""

        @before_run
        async def log_input(
            prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None
        ) -> str | Sequence[Any]:
            return f"logged: {prompt}"

        assert isinstance(log_input, AgentMiddleware)
        result = await log_input.before_run("test", None)
        assert result == "logged: test"

    async def test_before_run_can_block(self) -> None:
        """Test before_run can raise InputBlocked."""

        @before_run
        async def block_input(
            prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None
        ) -> str | Sequence[Any]:
            raise InputBlocked("Blocked!")

        with pytest.raises(InputBlocked):
            await block_input.before_run("test", None)

    async def test_before_run_other_methods_passthrough(self) -> None:
        """Test that other methods are passthroughs."""

        @before_run
        async def my_middleware(
            prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None
        ) -> str | Sequence[Any]:
            return prompt

        # Other methods should be passthrough
        assert await my_middleware.after_run("p", "out", None) == "out"
        assert await my_middleware.before_tool_call("t", {}, None) == {}
        assert await my_middleware.after_tool_call("t", {}, "r", None) == "r"
        assert await my_middleware.on_error(ValueError(), None) is None


class TestAfterRunDecorator:
    """Tests for after_run decorator."""

    async def test_after_run_basic(self) -> None:
        """Test basic after_run decorator."""

        @after_run
        async def modify_output(
            prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None
        ) -> Any:
            return f"modified: {output}"

        assert isinstance(modify_output, AgentMiddleware)
        result = await modify_output.after_run("prompt", "output", None)
        assert result == "modified: output"

    async def test_after_run_with_prompt_context(self) -> None:
        """Test after_run can access original prompt."""

        @after_run
        async def add_prompt_to_output(
            prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None
        ) -> Any:
            return {"prompt": prompt, "output": output}

        result = await add_prompt_to_output.after_run("original", "result", None)
        assert result == {"prompt": "original", "output": "result"}


class TestBeforeModelRequestDecorator:
    """Tests for before_model_request decorator."""

    async def test_before_model_request_basic(self) -> None:
        """Test basic before_model_request decorator."""

        @before_model_request
        async def log_messages(
            messages: list[Any], deps: None, ctx: ScopedContext | None
        ) -> list[Any]:
            return messages + [{"logged": True}]

        assert isinstance(log_messages, AgentMiddleware)
        result = await log_messages.before_model_request(cast(list[Any], [{"msg": 1}]), None)
        assert result == [{"msg": 1}, {"logged": True}]

    async def test_before_model_request_other_methods_passthrough(self) -> None:
        """Test that other methods are passthroughs for before_model_request."""

        @before_model_request
        async def my_middleware(
            messages: list[Any], deps: None, ctx: ScopedContext | None
        ) -> list[Any]:
            return messages

        # before_run should be passthrough
        assert await my_middleware.before_run("prompt", None) == "prompt"


class TestBeforeToolCallDecorator:
    """Tests for before_tool_call decorator."""

    async def test_before_tool_call_basic(self) -> None:
        """Test basic before_tool_call decorator."""

        @before_tool_call
        async def validate_tool(
            tool_name: str, tool_args: dict[str, Any], deps: None, ctx: ScopedContext | None
        ) -> dict[str, Any]:
            return {**tool_args, "validated": True}

        assert isinstance(validate_tool, AgentMiddleware)
        result = await validate_tool.before_tool_call("tool", {"arg": "val"}, None)
        assert result == {"arg": "val", "validated": True}

    async def test_before_tool_call_can_block(self) -> None:
        """Test before_tool_call can block tools."""

        @before_tool_call
        async def block_dangerous(
            tool_name: str, tool_args: dict[str, Any], deps: None, ctx: ScopedContext | None
        ) -> dict[str, Any]:
            if tool_name == "dangerous":
                raise ToolBlocked(tool_name, "Not allowed")
            return tool_args

        with pytest.raises(ToolBlocked) as exc_info:
            await block_dangerous.before_tool_call("dangerous", {}, None)
        assert exc_info.value.tool_name == "dangerous"


class TestAfterToolCallDecorator:
    """Tests for after_tool_call decorator."""

    async def test_after_tool_call_basic(self) -> None:
        """Test basic after_tool_call decorator."""

        @after_tool_call
        async def log_result(
            tool_name: str,
            tool_args: dict[str, Any],
            result: Any,
            deps: None,
            ctx: ScopedContext | None,
        ) -> Any:
            return {"tool": tool_name, "result": result}

        assert isinstance(log_result, AgentMiddleware)
        result = await log_result.after_tool_call("my_tool", {}, "success", None)
        assert result == {"tool": "my_tool", "result": "success"}

    async def test_after_tool_call_other_methods_passthrough(self) -> None:
        """Test that other methods are passthroughs for after_tool_call."""

        @after_tool_call
        async def my_middleware(
            tool_name: str,
            tool_args: dict[str, Any],
            result: Any,
            deps: None,
            ctx: ScopedContext | None,
        ) -> Any:
            return result

        # before_model_request should be passthrough
        messages: list[Any] = [{"msg": 1}]
        assert await my_middleware.before_model_request(messages, None) == messages


class TestOnErrorDecorator:
    """Tests for on_error decorator."""

    async def test_on_error_return_none(self) -> None:
        """Test on_error returning None re-raises original."""

        @on_error
        async def log_error(
            error: Exception, deps: None, ctx: ScopedContext | None
        ) -> Exception | None:
            return None

        assert isinstance(log_error, AgentMiddleware)
        result = await log_error.on_error(ValueError("test"), None)
        assert result is None

    async def test_on_error_return_different_exception(self) -> None:
        """Test on_error can return a different exception."""

        @on_error
        async def convert_error(
            error: Exception, deps: None, ctx: ScopedContext | None
        ) -> Exception | None:
            return RuntimeError("converted")

        result = await convert_error.on_error(ValueError("test"), None)
        assert isinstance(result, RuntimeError)
        assert str(result) == "converted"


class TestDecoratorWithDeps:
    """Tests for decorators with dependencies."""

    async def test_before_run_with_deps(self) -> None:
        """Test before_run decorator with dependencies."""

        @before_run
        async def use_deps(
            prompt: str | Sequence[Any],
            deps: dict[str, str] | None,
            ctx: ScopedContext | None,
        ) -> str | Sequence[Any]:
            if deps and "user" in deps:
                return f"[{deps['user']}] {prompt}"
            return prompt

        result = await use_deps.before_run("hello", {"user": "alice"})
        assert result == "[alice] hello"

    async def test_after_tool_call_with_deps(self) -> None:
        """Test after_tool_call decorator with dependencies."""

        @after_tool_call
        async def audit_tool(
            tool_name: str,
            tool_args: dict[str, Any],
            result: Any,
            deps: dict[str, str] | None,
            ctx: ScopedContext | None,
        ) -> Any:
            if deps and "audit" in deps:
                return {"result": result, "audited_by": deps["audit"]}
            return result

        result = await audit_tool.after_tool_call("tool", {}, "success", {"audit": "system"})
        assert result == {"result": "success", "audited_by": "system"}

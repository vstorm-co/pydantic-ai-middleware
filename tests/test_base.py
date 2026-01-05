"""Tests for base middleware module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai_middleware import AgentMiddleware


class TestAgentMiddleware:
    """Tests for AgentMiddleware base class."""

    async def test_before_run_default(self) -> None:
        """Test default before_run returns prompt unchanged."""
        middleware = ConcreteMiddleware()
        result = await middleware.before_run("test prompt", None)
        assert result == "test prompt"

    async def test_before_run_with_sequence(self) -> None:
        """Test before_run with sequence prompt."""
        middleware = ConcreteMiddleware()
        prompt: Sequence[Any] = ["part1", "part2"]
        result = await middleware.before_run(prompt, None)
        assert result == prompt

    async def test_after_run_default(self) -> None:
        """Test default after_run returns output unchanged."""
        middleware = ConcreteMiddleware()
        result = await middleware.after_run("prompt", "output", None)
        assert result == "output"

    async def test_before_model_request_default(self) -> None:
        """Test default before_model_request returns messages unchanged."""
        middleware = ConcreteMiddleware()
        messages: list[Any] = [{"role": "user", "content": "test"}]
        result = await middleware.before_model_request(messages, None)
        assert result == messages

    async def test_before_tool_call_default(self) -> None:
        """Test default before_tool_call returns args unchanged."""
        middleware = ConcreteMiddleware()
        args = {"param": "value"}
        result = await middleware.before_tool_call("my_tool", args, None)
        assert result == args

    async def test_after_tool_call_default(self) -> None:
        """Test default after_tool_call returns result unchanged."""
        middleware = ConcreteMiddleware()
        result = await middleware.after_tool_call("my_tool", {"param": "value"}, "result", None)
        assert result == "result"

    async def test_on_error_default(self) -> None:
        """Test default on_error returns None."""
        middleware = ConcreteMiddleware()
        error = ValueError("test error")
        result = await middleware.on_error(error, None)
        assert result is None

    async def test_with_deps(self) -> None:
        """Test middleware methods with dependencies."""

        class DepMiddleware(AgentMiddleware[dict[str, str]]):
            async def before_run(
                self, prompt: str | Sequence[Any], deps: dict[str, str] | None, ctx: Any = None
            ) -> str | Sequence[Any]:
                if deps and "prefix" in deps:
                    return f"{deps['prefix']}: {prompt}"
                return prompt

        middleware = DepMiddleware()
        deps = {"prefix": "User"}
        result = await middleware.before_run("hello", deps)
        assert result == "User: hello"


class ConcreteMiddleware(AgentMiddleware[None]):
    """Concrete implementation for testing."""

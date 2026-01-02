"""Decorator-based middleware for simple use cases."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from pydantic_ai.messages import ModelMessage

from .base import AgentMiddleware
from .context import ScopedContext

DepsT = TypeVar("DepsT")

# Type aliases for middleware functions
BeforeRunFunc = Callable[
    [str | Sequence[Any], DepsT | None, ScopedContext | None],
    Awaitable[str | Sequence[Any]],
]
AfterRunFunc = Callable[
    [str | Sequence[Any], Any, DepsT | None, ScopedContext | None],
    Awaitable[Any],
]
BeforeModelRequestFunc = Callable[
    [list[ModelMessage], DepsT | None, ScopedContext | None],
    Awaitable[list[ModelMessage]],
]
BeforeToolCallFunc = Callable[
    [str, dict[str, Any], DepsT | None, ScopedContext | None],
    Awaitable[dict[str, Any]],
]
AfterToolCallFunc = Callable[
    [str, dict[str, Any], Any, DepsT | None, ScopedContext | None],
    Awaitable[Any],
]
OnErrorFunc = Callable[
    [Exception, DepsT | None, ScopedContext | None],
    Awaitable[Exception | None],
]


class _FunctionMiddleware(AgentMiddleware[DepsT]):
    """Middleware that wraps a single function."""

    def __init__(
        self,
        *,
        before_run_func: BeforeRunFunc[DepsT] | None = None,
        after_run_func: AfterRunFunc[DepsT] | None = None,
        before_model_request_func: BeforeModelRequestFunc[DepsT] | None = None,
        before_tool_call_func: BeforeToolCallFunc[DepsT] | None = None,
        after_tool_call_func: AfterToolCallFunc[DepsT] | None = None,
        on_error_func: OnErrorFunc[DepsT] | None = None,
    ) -> None:
        self._before_run_func = before_run_func
        self._after_run_func = after_run_func
        self._before_model_request_func = before_model_request_func
        self._before_tool_call_func = before_tool_call_func
        self._after_tool_call_func = after_tool_call_func
        self._on_error_func = on_error_func

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        if self._before_run_func is not None:
            return await self._before_run_func(prompt, deps, ctx)
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        if self._after_run_func is not None:
            return await self._after_run_func(prompt, output, deps, ctx)
        return output

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> list[ModelMessage]:
        if self._before_model_request_func is not None:
            return await self._before_model_request_func(messages, deps, ctx)
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        if self._before_tool_call_func is not None:
            return await self._before_tool_call_func(tool_name, tool_args, deps, ctx)
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        if self._after_tool_call_func is not None:
            return await self._after_tool_call_func(tool_name, tool_args, result, deps, ctx)
        return result

    async def on_error(
        self,
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        if self._on_error_func is not None:
            return await self._on_error_func(error, deps, ctx)
        return None


def before_run(
    func: BeforeRunFunc[DepsT],
) -> AgentMiddleware[DepsT]:
    """Create middleware from a before_run function.

    Example:
        ```python
        @before_run
        async def log_input(prompt: str, deps, ctx) -> str:
            print(f"Input: {prompt}")
            ctx.set("logged", True)  # Store data in context
            return prompt
        ```
    """
    return _FunctionMiddleware(before_run_func=func)


def after_run(
    func: AfterRunFunc[DepsT],
) -> AgentMiddleware[DepsT]:
    """Create middleware from an after_run function.

    Example:
        ```python
        @after_run
        async def log_output(prompt: str, output, deps, ctx):
            print(f"Output: {output}")
            return output
        ```
    """
    return _FunctionMiddleware(after_run_func=func)


def before_model_request(
    func: BeforeModelRequestFunc[DepsT],
) -> AgentMiddleware[DepsT]:
    """Create middleware from a before_model_request function.

    Example:
        ```python
        @before_model_request
        async def log_messages(messages: list, deps, ctx) -> list:
            print(f"Sending {len(messages)} messages")
            return messages
        ```
    """
    return _FunctionMiddleware(before_model_request_func=func)


def before_tool_call(
    func: BeforeToolCallFunc[DepsT],
) -> AgentMiddleware[DepsT]:
    """Create middleware from a before_tool_call function.

    Example:
        ```python
        @before_tool_call
        async def validate_tool(tool_name: str, tool_args: dict, deps, ctx) -> dict:
            if tool_name == "dangerous_tool":
                raise ToolBlocked(tool_name, "Not allowed")
            return tool_args
        ```
    """
    return _FunctionMiddleware(before_tool_call_func=func)


def after_tool_call(
    func: AfterToolCallFunc[DepsT],
) -> AgentMiddleware[DepsT]:
    """Create middleware from an after_tool_call function.

    Example:
        ```python
        @after_tool_call
        async def log_tool_result(tool_name: str, tool_args: dict, result, deps, ctx):
            print(f"Tool {tool_name} returned: {result}")
            return result
        ```
    """
    return _FunctionMiddleware(after_tool_call_func=func)


def on_error(
    func: OnErrorFunc[DepsT],
) -> AgentMiddleware[DepsT]:
    """Create middleware from an on_error function.

    Example:
        ```python
        @on_error
        async def handle_error(error: Exception, deps, ctx) -> Exception | None:
            print(f"Error occurred: {error}")
            return None  # Re-raise original
        ```
    """
    return _FunctionMiddleware(on_error_func=func)

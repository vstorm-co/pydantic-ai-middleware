"""Decorator-based middleware for simple use cases."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar, overload

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
OnToolErrorFunc = Callable[
    [str, dict[str, Any], Exception, DepsT | None, ScopedContext | None],
    Awaitable[Exception | None],
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
        on_tool_error_func: OnToolErrorFunc[DepsT] | None = None,
        on_error_func: OnErrorFunc[DepsT] | None = None,
        tool_names: set[str] | None = None,
    ) -> None:
        self._before_run_func = before_run_func
        self._after_run_func = after_run_func
        self._before_model_request_func = before_model_request_func
        self._before_tool_call_func = before_tool_call_func
        self._after_tool_call_func = after_tool_call_func
        self._on_tool_error_func = on_tool_error_func
        self._on_error_func = on_error_func
        if tool_names is not None:
            self.tool_names = tool_names

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

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        if self._on_tool_error_func is not None:
            return await self._on_tool_error_func(tool_name, tool_args, error, deps, ctx)
        return None

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


@overload
def before_tool_call(
    func: BeforeToolCallFunc[DepsT],
) -> AgentMiddleware[DepsT]: ...


@overload
def before_tool_call(
    func: None = None,
    *,
    tools: set[str] | None = ...,
) -> Callable[[BeforeToolCallFunc[DepsT]], AgentMiddleware[DepsT]]: ...


def before_tool_call(
    func: BeforeToolCallFunc[DepsT] | None = None,
    *,
    tools: set[str] | None = None,
) -> AgentMiddleware[DepsT] | Callable[[BeforeToolCallFunc[DepsT]], AgentMiddleware[DepsT]]:
    """Create middleware from a before_tool_call function.

    Can be used as a plain decorator or with `tools` parameter:

    Example:
        ```python
        @before_tool_call
        async def validate_all(tool_name, tool_args, deps, ctx):
            return tool_args

        @before_tool_call(tools={"send_email"})
        async def validate_email(tool_name, tool_args, deps, ctx):
            return tool_args
        ```
    """
    if func is not None:
        # Used as @before_tool_call (no parentheses)
        return _FunctionMiddleware(before_tool_call_func=func, tool_names=tools)

    # Used as @before_tool_call(tools={"send_email"})
    def decorator(f: BeforeToolCallFunc[DepsT]) -> AgentMiddleware[DepsT]:
        return _FunctionMiddleware(before_tool_call_func=f, tool_names=tools)

    return decorator


@overload
def after_tool_call(
    func: AfterToolCallFunc[DepsT],
) -> AgentMiddleware[DepsT]: ...


@overload
def after_tool_call(
    func: None = None,
    *,
    tools: set[str] | None = ...,
) -> Callable[[AfterToolCallFunc[DepsT]], AgentMiddleware[DepsT]]: ...


def after_tool_call(
    func: AfterToolCallFunc[DepsT] | None = None,
    *,
    tools: set[str] | None = None,
) -> AgentMiddleware[DepsT] | Callable[[AfterToolCallFunc[DepsT]], AgentMiddleware[DepsT]]:
    """Create middleware from an after_tool_call function.

    Can be used as a plain decorator or with `tools` parameter:

    Example:
        ```python
        @after_tool_call
        async def log_all(tool_name, tool_args, result, deps, ctx):
            return result

        @after_tool_call(tools={"send_email"})
        async def log_email(tool_name, tool_args, result, deps, ctx):
            return result
        ```
    """
    if func is not None:
        return _FunctionMiddleware(after_tool_call_func=func, tool_names=tools)

    def decorator(f: AfterToolCallFunc[DepsT]) -> AgentMiddleware[DepsT]:
        return _FunctionMiddleware(after_tool_call_func=f, tool_names=tools)

    return decorator


@overload
def on_tool_error(
    func: OnToolErrorFunc[DepsT],
) -> AgentMiddleware[DepsT]: ...


@overload
def on_tool_error(
    func: None = None,
    *,
    tools: set[str] | None = ...,
) -> Callable[[OnToolErrorFunc[DepsT]], AgentMiddleware[DepsT]]: ...


def on_tool_error(
    func: OnToolErrorFunc[DepsT] | None = None,
    *,
    tools: set[str] | None = None,
) -> AgentMiddleware[DepsT] | Callable[[OnToolErrorFunc[DepsT]], AgentMiddleware[DepsT]]:
    """Create middleware from an on_tool_error function.

    Can be used as a plain decorator or with `tools` parameter:

    Example:
        ```python
        @on_tool_error
        async def handle_all_errors(tool_name, tool_args, error, deps, ctx):
            return None  # Re-raise original

        @on_tool_error(tools={"send_email"})
        async def handle_email_errors(tool_name, tool_args, error, deps, ctx):
            return RuntimeError("Email failed")
        ```
    """
    if func is not None:
        return _FunctionMiddleware(on_tool_error_func=func, tool_names=tools)

    def decorator(f: OnToolErrorFunc[DepsT]) -> AgentMiddleware[DepsT]:
        return _FunctionMiddleware(on_tool_error_func=f, tool_names=tools)

    return decorator


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

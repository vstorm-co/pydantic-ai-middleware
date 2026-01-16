"""Composable middleware chains."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar, overload

from pydantic_ai.messages import ModelMessage

from .base import AgentMiddleware
from .context import ScopedContext

DepsT = TypeVar("DepsT")


def _flatten(
    items: Iterable[AgentMiddleware[DepsT] | MiddlewareChain[DepsT]],
) -> list[AgentMiddleware[DepsT]]:
    flattened: list[AgentMiddleware[DepsT]] = []
    for item in items:
        if isinstance(item, MiddlewareChain):
            flattened.extend(item._middleware)
        elif isinstance(item, AgentMiddleware):
            flattened.append(item)
        else:
            raise TypeError(
                "MiddlewareChain items must be AgentMiddleware or MiddlewareChain, "
                f"got {type(item).__name__}"
            )
    return flattened


class MiddlewareChain(AgentMiddleware[DepsT], Generic[DepsT]):
    """A composable, ordered chain of middleware."""

    def __init__(
        self,
        middleware: Sequence[AgentMiddleware[DepsT] | MiddlewareChain[DepsT]] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self._middleware = _flatten(middleware or [])
        self._name = name or f"MiddlewareChain({len(self._middleware)})"

    @property
    def name(self) -> str:
        """Chain name for debugging/logging."""
        return self._name

    @property
    def middleware(self) -> list[AgentMiddleware[DepsT]]:
        """Return a copy of middleware list."""
        return list(self._middleware)

    def add(
        self, middleware: AgentMiddleware[DepsT] | MiddlewareChain[DepsT]
    ) -> MiddlewareChain[DepsT]:
        """Append middleware to the chain."""
        if isinstance(middleware, MiddlewareChain):
            self._middleware.extend(middleware._middleware)
        elif isinstance(middleware, AgentMiddleware):
            self._middleware.append(middleware)
        else:
            raise TypeError(
                "MiddlewareChain.add expects AgentMiddleware or MiddlewareChain, "
                f"got {type(middleware).__name__}"
            )
        return self

    def insert(
        self, index: int, middleware: AgentMiddleware[DepsT] | MiddlewareChain[DepsT]
    ) -> MiddlewareChain[DepsT]:
        """Insert middleware at a specific position."""
        if isinstance(middleware, MiddlewareChain):
            self._middleware[index:index] = middleware._middleware
        elif isinstance(middleware, AgentMiddleware):
            self._middleware.insert(index, middleware)
        else:
            raise TypeError(
                "MiddlewareChain.insert expects AgentMiddleware or MiddlewareChain, "
                f"got {type(middleware).__name__}"
            )
        return self

    def remove(self, middleware: AgentMiddleware[DepsT]) -> MiddlewareChain[DepsT]:
        """Remove middleware from the chain."""
        self._middleware.remove(middleware)
        return self

    def pop(self, index: int = -1) -> AgentMiddleware[DepsT]:
        """Remove and return middleware at index (default last)."""
        return self._middleware.pop(index)

    def replace(
        self,
        old: AgentMiddleware[DepsT],
        new: AgentMiddleware[DepsT] | MiddlewareChain[DepsT],
    ) -> MiddlewareChain[DepsT]:
        """Replace middleware with another middleware or chain."""
        index = self._middleware.index(old)
        if isinstance(new, MiddlewareChain):
            self._middleware[index : index + 1] = new._middleware
        elif isinstance(new, AgentMiddleware):
            self._middleware[index] = new
        else:
            raise TypeError(
                "MiddlewareChain.replace expects AgentMiddleware or MiddlewareChain, "
                f"got {type(new).__name__}"
            )
        return self

    def clear(self) -> MiddlewareChain[DepsT]:
        """Remove all middleware from the chain."""
        self._middleware.clear()
        return self

    def copy(self) -> MiddlewareChain[DepsT]:
        """Return a shallow copy of the chain."""
        return MiddlewareChain(self._middleware, name=self._name)

    def __add__(
        self, other: AgentMiddleware[DepsT] | MiddlewareChain[DepsT]
    ) -> MiddlewareChain[DepsT]:
        if isinstance(other, MiddlewareChain):
            return MiddlewareChain([*self._middleware, *other._middleware])
        if isinstance(other, AgentMiddleware):
            return MiddlewareChain([*self._middleware, other])
        raise TypeError(
            "MiddlewareChain + expects AgentMiddleware or MiddlewareChain, "
            f"got {type(other).__name__}"
        )

    def __iadd__(
        self, other: AgentMiddleware[DepsT] | MiddlewareChain[DepsT]
    ) -> MiddlewareChain[DepsT]:
        return self.add(other)

    def __len__(self) -> int:
        return len(self._middleware)

    def __bool__(self) -> bool:
        return bool(self._middleware)

    @overload
    def __getitem__(self, index: int) -> AgentMiddleware[DepsT]: ...

    @overload
    def __getitem__(self, index: slice) -> list[AgentMiddleware[DepsT]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]]:
        return self._middleware[index]

    def __iter__(self) -> Iterator[AgentMiddleware[DepsT]]:
        return iter(self._middleware)

    def __contains__(self, item: object) -> bool:
        return item in self._middleware

    def __repr__(self) -> str:
        return f"MiddlewareChain({self._middleware!r})"

    def __str__(self) -> str:
        if not self._middleware:
            return f"{self.name} (empty)"
        flow = " -> ".join(type(mw).__name__ for mw in self._middleware)
        return f"{self.name}: {flow}"

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        current_prompt: str | Sequence[Any] = prompt
        for mw in self._middleware:
            current_prompt = await mw.before_run(current_prompt, deps, ctx)
        return current_prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        current_output: Any = output
        for mw in reversed(self._middleware):
            current_output = await mw.after_run(prompt, current_output, deps, ctx)
        return current_output

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> list[ModelMessage]:
        current_messages = messages
        for mw in self._middleware:
            current_messages = await mw.before_model_request(current_messages, deps, ctx)
        return current_messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        current_args = tool_args
        for mw in self._middleware:
            current_args = await mw.before_tool_call(tool_name, current_args, deps, ctx)
        return current_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        current_result = result
        for mw in reversed(self._middleware):
            current_result = await mw.after_tool_call(
                tool_name, tool_args, current_result, deps, ctx
            )
        return current_result

    async def on_error(
        self,
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        for mw in self._middleware:
            handled = await mw.on_error(error, deps, ctx)
            if handled is not None:
                return handled
        return None

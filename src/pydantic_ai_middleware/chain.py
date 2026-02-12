"""Composable middleware chains."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from pydantic_ai.messages import ModelMessage

from .base import AgentMiddleware
from .context import ScopedContext

if TYPE_CHECKING:
    from .permissions import ToolPermissionResult

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
    """A composable, ordered chain of middleware.

    MiddlewareChain allows you to group multiple middleware into a reusable unit.
    When used in an agent, hooks are executed in sequence: before_* hooks run
    in order (first to last), while after_* hooks run in reverse order (last to first).

    Chains can be nested - adding a chain to another chain flattens the middleware
    into a single sequence.

    Example:
        ```python
        # Create a chain for security middleware
        security_chain = MiddlewareChain([
            AuthMiddleware(),
            RateLimitMiddleware(),
        ], name="security")

        # Use with an agent
        agent = MiddlewareAgent(
            agent=base_agent,
            middleware=[security_chain, LoggingMiddleware()],
        )
        ```
    """

    def __init__(
        self,
        middleware: Sequence[AgentMiddleware[DepsT] | MiddlewareChain[DepsT]] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Initialize a middleware chain.

        Args:
            middleware: Sequence of middleware or chains to include.
                Chains are flattened into their constituent middleware.
            name: Optional name for the chain (useful for debugging/logging).
                Defaults to "MiddlewareChain(N)" where N is the count.
        """
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
        """Append middleware to the end of the chain.

        Args:
            middleware: Middleware or chain to append. If a chain is provided,
                its middleware are flattened into this chain.

        Returns:
            Self for method chaining.

        Raises:
            TypeError: If middleware is not an AgentMiddleware or MiddlewareChain.
        """
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
        """Insert middleware at a specific position in the chain.

        Args:
            index: Position to insert at (0-based).
            middleware: Middleware or chain to insert. If a chain is provided,
                its middleware are flattened and inserted at the position.

        Returns:
            Self for method chaining.

        Raises:
            TypeError: If middleware is not an AgentMiddleware or MiddlewareChain.
        """
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
        """Remove a specific middleware from the chain.

        Args:
            middleware: The middleware instance to remove.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If middleware is not in the chain.
        """
        self._middleware.remove(middleware)
        return self

    def pop(self, index: int = -1) -> AgentMiddleware[DepsT]:
        """Remove and return middleware at the given index.

        Args:
            index: Position to pop from (default: -1, the last item).

        Returns:
            The removed middleware instance.

        Raises:
            IndexError: If the chain is empty or index is out of range.
        """
        return self._middleware.pop(index)

    def replace(
        self,
        old: AgentMiddleware[DepsT],
        new: AgentMiddleware[DepsT] | MiddlewareChain[DepsT],
    ) -> MiddlewareChain[DepsT]:
        """Replace a middleware with another middleware or chain.

        Args:
            old: The middleware instance to replace.
            new: Middleware or chain to replace with. If a chain is provided,
                its middleware are flattened into the position.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If old middleware is not in the chain.
            TypeError: If new is not an AgentMiddleware or MiddlewareChain.
        """
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
        """Remove all middleware from the chain.

        Returns:
            Self for method chaining.
        """
        self._middleware.clear()
        return self

    def copy(self) -> MiddlewareChain[DepsT]:
        """Return a shallow copy of the chain.

        Returns:
            A new MiddlewareChain with the same middleware and name.
        """
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
    ) -> dict[str, Any] | ToolPermissionResult:
        current_args: dict[str, Any] = tool_args
        for mw in self._middleware:
            if not mw._should_handle_tool(tool_name):
                continue
            result = await mw.before_tool_call(tool_name, current_args, deps, ctx)
            if isinstance(result, dict):
                current_args = result
            else:
                # ToolPermissionResult - short-circuit the chain
                return result
        return current_args

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        for mw in self._middleware:
            if not mw._should_handle_tool(tool_name):
                continue
            handled = await mw.on_tool_error(tool_name, tool_args, error, deps, ctx)
            if handled is not None:
                return handled
        return None

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
            if not mw._should_handle_tool(tool_name):
                continue
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

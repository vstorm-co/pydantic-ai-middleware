"""Conditional middleware for branching execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic_ai.messages import ModelMessage

from .base import AgentMiddleware, DepsT
from .context import ScopedContext

if TYPE_CHECKING:
    from .permissions import ToolPermissionResult

CtxT = TypeVar("CtxT", bound=ScopedContext | None)
"""Type variable for condition predicate context."""

Predicate = Callable[[CtxT], bool]
"""Predicate function that takes a scoped context and returns bool."""

PredicateFactory = Callable[..., Predicate[ScopedContext | None]]
"""Factory for predicate functions."""


class ConditionalMiddleware(AgentMiddleware[DepsT], Generic[DepsT]):
    """Route to different middleware based on a runtime condition.

    The condition is evaluated for each hook invocation, allowing
    different middleware to handle different hooks. This wraps standard
    middleware instances or a middleware pipeline (list of middleware).

    Example:
        ```python
        def is_after_run(ctx: ScopedContext | None) -> bool:
            return ctx is not None and ctx.current_hook == HookType.AFTER_RUN

        middleware = ConditionalMiddleware(
            condition=is_after_run,
            when_true=AfterRunMiddleware(),
            when_false=OtherMiddleware(),
        )
        ```
    """

    def __init__(
        self,
        condition: Predicate[ScopedContext | None],
        when_true: AgentMiddleware[DepsT] | Sequence[AgentMiddleware[DepsT]],
        when_false: AgentMiddleware[DepsT] | Sequence[AgentMiddleware[DepsT]] | None = None,
    ) -> None:
        """Initialize conditional middleware.

        Args:
            condition: Predicate function that receives ScopedContext.
            when_true: Middleware to execute when condition is True.
                Can be a single middleware or a sequence (executed in order).
            when_false: Middleware to execute when condition is False.
                Can be a single middleware, sequence, or None (pass-through).

        Raises:
            TypeError: If middleware arguments are invalid types.
        """
        self.condition = condition
        self.when_true = self._normalize_middleware(when_true)
        self.when_false = self._normalize_middleware(when_false) if when_false else None

    def _normalize_middleware(
        self, mw: AgentMiddleware[DepsT] | Sequence[AgentMiddleware[DepsT]]
    ) -> list[AgentMiddleware[DepsT]]:
        """Normalize middleware to a list."""
        if isinstance(mw, Sequence) and not isinstance(mw, AgentMiddleware):
            return self._validate_middleware_sequence(mw)
        return [self._validate_middleware_item(mw)]

    def _validate_middleware_item(self, item: Any, context: str = "") -> AgentMiddleware[Any]:
        """Validate that an item is a middleware instance."""
        if not isinstance(item, AgentMiddleware):
            msg = f"Expected AgentMiddleware{context}, got {type(item).__name__}"
            raise TypeError(msg)
        return item

    def _validate_middleware_sequence(self, items: Sequence[Any]) -> list[AgentMiddleware[Any]]:
        """Validate and flatten a sequence of middleware."""
        result: list[AgentMiddleware[Any]] = []
        for item in items:
            if not isinstance(item, AgentMiddleware):
                raise TypeError(
                    f"Expected AgentMiddleware instances in sequence, got {type(item).__name__}"
                )
            result.append(item)
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConditionalMiddleware(when_true={self.when_true!r}, when_false={self.when_false!r})"
        )

    def _select(self, ctx: ScopedContext | None) -> list[AgentMiddleware[DepsT]] | None:
        """Select middleware based on condition."""
        if self.condition(ctx):
            return self.when_true
        return self.when_false

    async def _run_before(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> str | Sequence[Any]:
        result = prompt
        for mw in middleware:
            result = await mw.before_run(result, deps, ctx)
        return result

    async def _run_after(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> Any:
        result = output
        for mw in reversed(middleware):
            result = await mw.after_run(prompt, result, deps, ctx)
        return result

    async def _run_before_model_request(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        messages: list[ModelMessage],
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> list[ModelMessage]:
        result = messages
        for mw in middleware:
            result = await mw.before_model_request(result, deps, ctx)
        return result

    async def _run_before_tool_call(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> dict[str, Any] | ToolPermissionResult:
        current_args: dict[str, Any] = tool_args
        for mw in middleware:
            if not mw._should_handle_tool(tool_name):
                continue
            result = await mw.before_tool_call(tool_name, current_args, deps, ctx)
            if isinstance(result, dict):
                current_args = result
            else:
                return result
        return current_args

    async def _run_after_tool_call(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> Any:
        output = result
        for mw in reversed(middleware):
            if not mw._should_handle_tool(tool_name):
                continue
            output = await mw.after_tool_call(tool_name, tool_args, output, deps, ctx)
        return output

    async def _run_on_tool_error(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> Exception | None:
        for mw in middleware:
            if not mw._should_handle_tool(tool_name):
                continue
            handled = await mw.on_tool_error(tool_name, tool_args, error, deps, ctx)
            if handled is not None:
                return handled
        return None

    async def _run_on_error(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None,
    ) -> Exception | None:
        for mw in middleware:
            handled = await mw.on_error(error, deps, ctx)
            if handled is not None:
                return handled
        return None

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        """Execute selected middleware's before_run hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly.

        Args:
            prompt: The user prompt to process.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            The (possibly modified) prompt from the selected middleware.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_before(middleware, prompt, deps, ctx)
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Execute selected middleware's after_run hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly. Middleware is executed in reverse order.

        Args:
            prompt: The original user prompt.
            output: The agent output to process.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            The (possibly modified) output from the selected middleware.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_after(middleware, prompt, output, deps, ctx)
        return output

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> list[ModelMessage]:
        """Execute selected middleware's before_model_request hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly.

        Args:
            messages: The messages to send to the model.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            The (possibly modified) messages from the selected middleware.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_before_model_request(middleware, messages, deps, ctx)
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any] | ToolPermissionResult:
        """Execute selected middleware's before_tool_call hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly.

        Args:
            tool_name: The name of the tool being called.
            tool_args: The arguments to the tool.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            The (possibly modified) tool arguments from the selected middleware.

        Raises:
            ToolBlocked: If any selected middleware blocks the tool call.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_before_tool_call(middleware, tool_name, tool_args, deps, ctx)
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Execute selected middleware's after_tool_call hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly. Middleware is executed in reverse order.

        Args:
            tool_name: The name of the tool that was called.
            tool_args: The arguments that were passed to the tool.
            result: The result from the tool.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            The (possibly modified) result from the selected middleware.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_after_tool_call(
                middleware, tool_name, tool_args, result, deps, ctx
            )
        return result

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        """Execute selected middleware's on_tool_error hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly.

        Args:
            tool_name: The name of the tool that failed.
            tool_args: The arguments that were passed to the tool.
            error: The exception raised by the tool.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            A different exception to raise, or None to re-raise the original.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_on_tool_error(middleware, tool_name, tool_args, error, deps, ctx)
        return None

    async def on_error(
        self,
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        """Execute selected middleware's on_error hook.

        Evaluates the condition and routes to `when_true` or `when_false`
        middleware pipeline accordingly.

        Args:
            error: The exception that occurred.
            deps: The agent dependencies.
            ctx: Scoped context for condition evaluation and data sharing.

        Returns:
            A different exception to raise, or None to re-raise the original.
        """
        middleware = self._select(ctx)
        if middleware:
            return await self._run_on_error(middleware, error, deps, ctx)
        return None

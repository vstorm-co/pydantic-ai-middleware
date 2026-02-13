"""Middleware toolset for intercepting tool calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset
from typing_extensions import Self

from ._timeout import call_with_timeout
from .exceptions import ToolBlocked
from .permissions import PermissionHandler, ToolDecision, ToolPermissionResult

if TYPE_CHECKING:
    from .base import AgentMiddleware
    from .context import MiddlewareContext

DepsT = TypeVar("DepsT")


@dataclass
class MiddlewareToolset(WrapperToolset[DepsT], Generic[DepsT]):
    """Toolset that applies middleware to tool calls.

    This toolset wraps another toolset and applies middleware
    before and after each tool call.
    """

    wrapped: AbstractToolset[DepsT]
    middleware: list[AgentMiddleware[DepsT]] = field(default_factory=list)
    ctx: MiddlewareContext | None = None
    permission_handler: PermissionHandler | None = None

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f"MiddlewareToolset({self.wrapped.label})"

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[DepsT]) -> dict[str, ToolsetTool[DepsT]]:
        return await self.wrapped.get_tools(ctx)

    async def _process_permission_result(
        self,
        result: dict[str, Any] | ToolPermissionResult,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Process the return value of before_tool_call.

        Handles both dict returns (backwards compatible) and
        ToolPermissionResult returns (new permission protocol).

        Args:
            result: The return value from before_tool_call.
            tool_name: The name of the tool being called.
            tool_args: The original tool arguments.

        Returns:
            The final tool arguments to use.

        Raises:
            ToolBlocked: If the permission decision is DENY or ASK without handler.
        """
        if isinstance(result, dict):
            return result

        if result.decision == ToolDecision.ALLOW:
            return result.modified_args if result.modified_args is not None else tool_args

        if result.decision == ToolDecision.DENY:
            raise ToolBlocked(tool_name, result.reason or "Permission denied")

        # ASK decision
        if self.permission_handler is not None:
            allowed = await self.permission_handler(tool_name, tool_args, result.reason)
            if allowed:
                return result.modified_args if result.modified_args is not None else tool_args
            raise ToolBlocked(tool_name, result.reason or "Permission denied by handler")

        raise ToolBlocked(
            tool_name, result.reason or "Permission required but no handler configured"
        )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[DepsT],
        tool: ToolsetTool[DepsT],
    ) -> Any:
        """Call a tool with middleware hooks."""
        from .context import HookType

        deps = ctx.deps if ctx else None

        # Get scoped contexts for tool hooks
        before_tool_ctx = self.ctx.for_hook(HookType.BEFORE_TOOL_CALL) if self.ctx else None
        after_tool_ctx = self.ctx.for_hook(HookType.AFTER_TOOL_CALL) if self.ctx else None

        # Apply before_tool_call middleware (with tool name filtering and timeout)
        current_args = tool_args
        for mw in self.middleware:
            if not mw._should_handle_tool(name):
                continue
            mw_name = type(mw).__name__
            raw_result = await call_with_timeout(
                mw.before_tool_call(name, current_args, deps, before_tool_ctx),
                mw.timeout,
                mw_name,
                "before_tool_call",
            )
            current_args = await self._process_permission_result(raw_result, name, current_args)

        # Execute the tool with on_tool_error handling
        try:
            result = await self.wrapped.call_tool(name, current_args, ctx, tool)
        except Exception as tool_error:
            on_tool_error_ctx = (
                self.ctx.for_hook(HookType.ON_TOOL_ERROR) if self.ctx else None
            )
            for mw in self.middleware:
                if not mw._should_handle_tool(name):
                    continue
                mw_name = type(mw).__name__
                handled = await call_with_timeout(
                    mw.on_tool_error(name, current_args, tool_error, deps, on_tool_error_ctx),
                    mw.timeout,
                    mw_name,
                    "on_tool_error",
                )
                if handled is not None:
                    raise handled from tool_error
            raise

        # Apply after_tool_call middleware (in reverse order, with tool name filtering and timeout)
        for mw in reversed(self.middleware):
            if not mw._should_handle_tool(name):
                continue
            mw_name = type(mw).__name__
            result = await call_with_timeout(
                mw.after_tool_call(name, current_args, result, deps, after_tool_ctx),
                mw.timeout,
                mw_name,
                "after_tool_call",
            )

        return result

    def apply(self, visitor: Callable[[AbstractToolset[DepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[DepsT]], AbstractToolset[DepsT]]
    ) -> AbstractToolset[DepsT]:
        return replace(self, wrapped=self.wrapped.visit_and_replace(visitor))

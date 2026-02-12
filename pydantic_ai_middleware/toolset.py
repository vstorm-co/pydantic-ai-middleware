"""Middleware toolset for intercepting tool calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset
from typing_extensions import Self

if TYPE_CHECKING:
    from .base import AgentMiddleware

DepsT = TypeVar("DepsT")


@dataclass
class MiddlewareToolset(WrapperToolset[DepsT], Generic[DepsT]):
    """Toolset that applies middleware to tool calls.

    This toolset wraps another toolset and applies middleware
    before and after each tool call.
    """

    wrapped: AbstractToolset[DepsT]
    middleware: list[AgentMiddleware[DepsT]] = field(default_factory=list)

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

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[DepsT],
        tool: ToolsetTool[DepsT],
    ) -> Any:
        """Call a tool with middleware hooks."""
        deps = ctx.deps if ctx else None

        # Apply before_tool_call middleware
        current_args = tool_args
        for mw in self.middleware:
            current_args = await mw.before_tool_call(name, current_args, deps)

        # Execute the tool
        result = await self.wrapped.call_tool(name, current_args, ctx, tool)

        # Apply after_tool_call middleware (in reverse order)
        for mw in reversed(self.middleware):
            result = await mw.after_tool_call(name, current_args, result, deps)

        return result

    def apply(self, visitor: Callable[[AbstractToolset[DepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[DepsT]], AbstractToolset[DepsT]]
    ) -> AbstractToolset[DepsT]:
        return replace(self, wrapped=self.wrapped.visit_and_replace(visitor))

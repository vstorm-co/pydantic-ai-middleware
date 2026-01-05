"""Base middleware class for pydantic-ai-middleware."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic_ai.messages import ModelMessage

if TYPE_CHECKING:
    from .context import ScopedContext

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


class AgentMiddleware(ABC, Generic[DepsT]):
    """Base middleware class.

    Inherit and override only the methods you need.
    Middleware hooks are called in order for before_* methods
    and in reverse order for after_* methods.
    """

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        """Called before the agent runs.

        Args:
            prompt: The user prompt.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing (write: before_run, read: config/metadata).

        Returns:
            The (possibly modified) prompt.

        Raises:
            InputBlocked: To block the input.
        """
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Called after the agent finishes.

        Args:
            prompt: The original user prompt.
            output: The agent output.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing (write: after_run, read: all previous).

        Returns:
            The (possibly modified) output.
        """
        return output

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> list[ModelMessage]:
        """Called before each model request.

        Args:
            messages: The messages to send to the model.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing (write: before_model_request, read: before_run).

        Returns:
            The (possibly modified) messages.
        """
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        """Called before a tool is called.

        Args:
            tool_name: The name of the tool being called.
            tool_args: The arguments to the tool.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing
            (write: before_tool_call, read: before_run, before_model_request).

        Returns:
            The (possibly modified) tool arguments.

        Raises:
            ToolBlocked: To block the tool call.
        """
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        """Called after a tool is called.

        Args:
            tool_name: The name of the tool that was called.
            tool_args: The arguments that were passed to the tool.
            result: The result from the tool.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing (write: after_tool_call, read: all before_* hooks).

        Returns:
            The (possibly modified) result.
        """
        return result

    async def on_error(
        self,
        error: Exception,
        deps: DepsT | None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        """Called when an error occurs.

        Args:
            error: The exception that occurred.
            deps: The agent dependencies.
            ctx: Scoped context for data sharing (write: on_error, read: all hooks).

        Returns:
            A different exception to raise, or None to re-raise the original.
        """
        return None

"""Middleware agent that wraps another agent with middleware hooks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from pydantic_ai import messages as _messages
from pydantic_ai import models
from pydantic_ai import usage as _usage
from pydantic_ai._utils import UNSET, Unset
from pydantic_ai.agent.abstract import AbstractAgent, EventStreamHandler, Instructions
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    BuiltinToolFunc,
    DeferredToolResults,
    Tool,
    ToolFuncEither,
)
from pydantic_ai.toolsets import AbstractToolset

from ._timeout import call_with_timeout
from .base import AgentMiddleware
from .context import HookType, MiddlewareContext
from .permissions import PermissionHandler
from .toolset import MiddlewareToolset


class MiddlewareAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Agent that wraps another agent and applies middleware hooks.

    Middleware is applied in order for before_* hooks and in reverse
    order for after_* hooks, similar to ASGI/WSGI middleware.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        middleware: list[AgentMiddleware[AgentDepsT]] | None = None,
        context: MiddlewareContext | None = None,
        permission_handler: PermissionHandler | None = None,
    ) -> None:
        """Initialize the middleware agent.

        Args:
            agent: The agent to wrap.
            middleware: List of middleware to apply.
            context: MiddlewareContext instance for sharing data
                     between hooks. If not provided, context features are disabled.
            permission_handler: Callback for handling ASK permission decisions
                     from before_tool_call. Called with (tool_name, tool_args, reason)
                     and should return True to allow or False to deny.
        """
        self._wrapped = agent
        self._middleware = middleware or []
        self._context = context
        self._permission_handler = permission_handler

    @property
    def wrapped(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        """The wrapped agent."""
        return self._wrapped

    @property
    def middleware(self) -> list[AgentMiddleware[AgentDepsT]]:
        """The middleware list."""
        return self._middleware

    @property
    def context(self) -> MiddlewareContext | None:
        """The middleware context for sharing data between hooks."""
        return self._context

    @property
    def model(self) -> models.Model | models.KnownModelName | str | None:
        return self._wrapped.model

    @property
    def name(self) -> str | None:
        return self._wrapped.name

    @name.setter
    def name(self, value: str | None) -> None:
        self._wrapped.name = value

    @property
    def deps_type(self) -> type:
        return self._wrapped.deps_type

    @property
    def output_type(self) -> OutputSpec[OutputDataT]:
        return self._wrapped.output_type

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        return self._wrapped.event_stream_handler

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        return self._wrapped.toolsets

    async def __aenter__(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        return await self._wrapped.__aenter__()

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self._wrapped.__aexit__(*args)

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[Any]:
        """Run the agent with middleware hooks applied.

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type for this run.
            message_history: History of the conversation so far.
            deferred_tool_results: Results for deferred tool calls.
            model: Optional model to use for this run.
            instructions: Optional additional instructions.
            deps: Optional dependencies for this run.
            model_settings: Optional settings for the model request.
            usage_limits: Optional limits on model requests or tokens.
            usage: Optional usage to start with.
            infer_name: Whether to infer agent name from call frame.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools.
            event_stream_handler: Optional handler for stream events.

        Returns:
            The result of the agent run.
        """
        if self._context is not None:
            ctx: MiddlewareContext = self._context
            # Reset context state from previous runs
            ctx.reset()
            # Set run metadata
            ctx.set_metadata("user_prompt", user_prompt)
        else:
            ctx = None

        try:
            # Apply before_run middleware (with timeout)
            current_prompt: str | Sequence[Any] | None = user_prompt
            before_run_ctx = ctx.for_hook(HookType.BEFORE_RUN) if ctx else None
            for mw in self._middleware:
                if current_prompt is not None:
                    mw_name = type(mw).__name__
                    current_prompt = await call_with_timeout(
                        mw.before_run(current_prompt, deps, before_run_ctx),
                        mw.timeout,
                        mw_name,
                        "before_run",
                    )

            # Store transformed prompt in metadata
            if ctx:
                ctx.set_metadata("transformed_prompt", current_prompt)

            # Wrap toolsets with middleware
            middleware_toolsets: list[AbstractToolset[AgentDepsT]] = []
            if toolsets:
                for ts in toolsets:
                    middleware_toolsets.append(
                        MiddlewareToolset(
                            wrapped=ts,
                            middleware=self._middleware,
                            ctx=ctx,
                            permission_handler=self._permission_handler,
                        )
                    )

            # Run the wrapped agent
            result = await self._wrapped.run(
                current_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=middleware_toolsets if middleware_toolsets else None,
                builtin_tools=builtin_tools,
                event_stream_handler=event_stream_handler,
            )

            # Apply after_run middleware (in reverse order, with timeout)
            output = result.output
            after_run_ctx = ctx.for_hook(HookType.AFTER_RUN) if ctx else None
            for mw in reversed(self._middleware):
                if current_prompt is not None:
                    mw_name = type(mw).__name__
                    output = await call_with_timeout(
                        mw.after_run(current_prompt, output, deps, after_run_ctx),
                        mw.timeout,
                        mw_name,
                        "after_run",
                    )

            # Store final output in metadata
            if ctx:
                ctx.set_metadata("final_output", output)

            # Return result with possibly modified output
            return _create_result_with_output(result, output)

        except Exception as e:
            # Apply on_error middleware (with timeout)
            on_error_ctx = ctx.for_hook(HookType.ON_ERROR) if ctx else None
            for mw in self._middleware:
                mw_name = type(mw).__name__
                handled = await call_with_timeout(
                    mw.on_error(e, deps, on_error_ctx),
                    mw.timeout,
                    mw_name,
                    "on_error",
                )
                if handled is not None:
                    raise handled from e
            raise

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """Iterate over agent execution with middleware applied."""
        # Use provided context or None
        if self._context is not None:
            ctx: MiddlewareContext = self._context
            # Reset context state from previous runs
            ctx.reset()
            # Set run metadata
            ctx.set_metadata("user_prompt", user_prompt)
        else:
            ctx = None

        # Apply before_run middleware (with timeout)
        current_prompt: str | Sequence[Any] | None = user_prompt
        before_run_ctx = ctx.for_hook(HookType.BEFORE_RUN) if ctx else None
        for mw in self._middleware:
            if current_prompt is not None:
                mw_name = type(mw).__name__
                current_prompt = await call_with_timeout(
                    mw.before_run(current_prompt, deps, before_run_ctx),
                    mw.timeout,
                    mw_name,
                    "before_run",
                )

        # Store transformed prompt in metadata
        if ctx:
            ctx.set_metadata("transformed_prompt", current_prompt)

        # Wrap toolsets with middleware
        middleware_toolsets: list[AbstractToolset[AgentDepsT]] = []
        if toolsets:
            for ts in toolsets:
                middleware_toolsets.append(
                    MiddlewareToolset(
                        wrapped=ts,
                        middleware=self._middleware,
                        ctx=ctx,
                        permission_handler=self._permission_handler,
                    )
                )

        async with self._wrapped.iter(
            user_prompt=current_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=middleware_toolsets if middleware_toolsets else None,
            builtin_tools=builtin_tools,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | Unset = UNSET,
        deps: AgentDepsT | Unset = UNSET,
        model: models.Model | models.KnownModelName | str | Unset = UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | Unset = UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset = UNSET,
        instructions: Instructions[AgentDepsT] | Unset = UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent settings."""
        with self._wrapped.override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            instructions=instructions,
        ):
            yield


def _create_result_with_output(result: AgentRunResult[Any], output: Any) -> AgentRunResult[Any]:
    """Create a new AgentRunResult with a different output value."""
    # AgentRunResult is a dataclass, so we create a new instance
    return AgentRunResult(
        output=output,
        _output_tool_name=result._output_tool_name,  # type: ignore[reportPrivateUsage]
        _state=result._state,  # type: ignore[reportPrivateUsage]
        _new_message_index=result._new_message_index,  # type: ignore[reportPrivateUsage]
        _traceparent_value=result._traceparent_value,  # type: ignore[reportPrivateUsage]
    )

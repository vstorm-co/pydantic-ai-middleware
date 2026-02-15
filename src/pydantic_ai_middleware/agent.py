"""Middleware agent that wraps another agent with middleware hooks."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from pydantic_ai import messages as _messages
from pydantic_ai import models
from pydantic_ai import usage as _usage
from pydantic_ai.agent.abstract import (
    AbstractAgent,
    AgentMetadata,
    EventStreamHandler,
    Instructions,
)
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    BuiltinToolFunc,
    DeferredToolResults,
    RunContext,
)
from pydantic_ai.toolsets import AbstractToolset

from ._timeout import call_with_timeout
from .base import AgentMiddleware
from .context import HookType, MiddlewareContext, ScopedContext
from .permissions import PermissionHandler
from .toolset import MiddlewareToolset


def _create_bmr_processor(
    middleware: list[AgentMiddleware[Any]],
    ctx: MiddlewareContext | None,
) -> Any:
    """Create a history processor that bridges to before_model_request middleware.

    pydantic-ai calls history processors in ``ModelRequestNode._prepare_request()``
    before every model request — both in ``run()`` and ``iter()`` mode.  We use this
    to invoke the ``before_model_request`` hook on each middleware.
    """
    bmr_ctx: ScopedContext | None = ctx.for_hook(HookType.BEFORE_MODEL_REQUEST) if ctx else None

    async def _processor(
        run_context: RunContext[Any], messages: list[_messages.ModelMessage]
    ) -> list[_messages.ModelMessage]:
        current = messages
        for mw in middleware:
            mw_name = type(mw).__name__
            current = await call_with_timeout(
                mw.before_model_request(current, run_context.deps, bmr_ctx),
                mw.timeout,
                mw_name,
                "before_model_request",
            )
        return current

    return _processor


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
        metadata: AgentMetadata[AgentDepsT] | None = None,
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
            metadata: Optional metadata for the agent run.
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

            # Wrap toolsets with middleware — always wrap all toolsets
            # (both the agent's own + any explicitly passed) so middleware
            # intercepts every tool call.
            all_toolsets = list(toolsets) if toolsets else list(self._wrapped.toolsets)
            middleware_toolsets: list[AbstractToolset[AgentDepsT]] = []
            for ts in all_toolsets:
                middleware_toolsets.append(
                    MiddlewareToolset(
                        wrapped=ts,
                        middleware=self._middleware,
                        ctx=ctx,
                        permission_handler=self._permission_handler,
                    )
                )

            # Inject before_model_request middleware as a history processor.
            # pydantic-ai calls history processors before every model request,
            # which is exactly the hook point we need.
            bmr_processor = _create_bmr_processor(self._middleware, ctx)
            original_processors = getattr(self._wrapped, "history_processors", [])
            self._wrapped.history_processors = list(original_processors) + [bmr_processor]  # type: ignore[union-attr]

            try:
                # Use override() to REPLACE the agent's toolsets (not add to them),
                # then call run() without passing toolsets= to avoid duplicates.
                with self._wrapped.override(toolsets=middleware_toolsets):
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
                        metadata=metadata,
                        infer_name=infer_name,
                        builtin_tools=builtin_tools,
                        event_stream_handler=event_stream_handler,
                    )
            finally:
                self._wrapped.history_processors = original_processors  # type: ignore[union-attr]

            # Store run usage in metadata for middleware access (e.g. cost tracking)
            if ctx:
                ctx.set_metadata("run_usage", result.usage())

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
        metadata: AgentMetadata[AgentDepsT] | None = None,
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

        # Wrap toolsets with middleware — always wrap all toolsets
        # (both the agent's own + any explicitly passed) so middleware
        # intercepts every tool call.
        all_toolsets = list(toolsets) if toolsets else list(self._wrapped.toolsets)
        middleware_toolsets: list[AbstractToolset[AgentDepsT]] = []
        for ts in all_toolsets:
            middleware_toolsets.append(
                MiddlewareToolset(
                    wrapped=ts,
                    middleware=self._middleware,
                    ctx=ctx,
                    permission_handler=self._permission_handler,
                )
            )

        # Inject before_model_request middleware as a history processor.
        bmr_processor = _create_bmr_processor(self._middleware, ctx)
        original_processors = getattr(self._wrapped, "history_processors", [])
        self._wrapped.history_processors = list(original_processors) + [bmr_processor]  # type: ignore[union-attr]

        try:
            # Use override() to REPLACE the agent's toolsets (not add to them),
            # then call iter() without passing toolsets= to avoid duplicates.
            with self._wrapped.override(toolsets=middleware_toolsets):
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
                    metadata=metadata,
                    infer_name=infer_name,
                    builtin_tools=builtin_tools,
                ) as run:
                    yield run
        finally:
            self._wrapped.history_processors = original_processors  # type: ignore[union-attr]

    @contextmanager
    def override(self, **kwargs: Any) -> Iterator[None]:
        """Context manager to temporarily override agent settings.

        Accepts the same keyword arguments as the wrapped agent's override method
        (name, deps, model, toolsets, tools, instructions).
        """
        with self._wrapped.override(**kwargs):
            yield


def _create_result_with_output(result: AgentRunResult[Any], output: Any) -> AgentRunResult[Any]:
    """Create a new AgentRunResult with a different output value."""
    # Use dataclasses.replace() to avoid accessing private attributes directly
    return dataclasses.replace(result, output=output)

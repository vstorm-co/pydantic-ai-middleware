"""Tests for MiddlewareAgent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    InputBlocked,
    MiddlewareAgent,
    MiddlewareContext,
    ScopedContext,
)


class LoggingMiddleware(AgentMiddleware[None]):
    """Middleware that logs agent lifecycle events."""

    def __init__(self) -> None:
        self.before_run_calls: list[str | Sequence[Any]] = []
        self.after_run_calls: list[tuple[str | Sequence[Any], Any]] = []
        self.errors: list[Exception] = []

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        self.before_run_calls.append(prompt)
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        self.after_run_calls.append((prompt, output))
        return output

    async def on_error(
        self, error: Exception, deps: None, ctx: ScopedContext | None = None
    ) -> Exception | None:
        self.errors.append(error)
        return None


class ModifyingMiddleware(AgentMiddleware[None]):
    """Middleware that modifies prompts and outputs."""

    def __init__(self, prefix: str = "modified") -> None:
        self.prefix = prefix

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        if isinstance(prompt, str):
            return f"{self.prefix}: {prompt}"
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        return f"{self.prefix} output: {output}"


class BlockingMiddleware(AgentMiddleware[None]):
    """Middleware that blocks certain inputs."""

    def __init__(self, blocked_words: set[str]) -> None:
        self.blocked_words = blocked_words

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        if isinstance(prompt, str):
            for word in self.blocked_words:
                if word in prompt.lower():
                    raise InputBlocked(f"Blocked word: {word}")
        return prompt


class ErrorHandlingMiddleware(AgentMiddleware[None]):
    """Middleware that handles errors."""

    def __init__(self, convert_to: type[Exception] | None = None) -> None:
        self.convert_to = convert_to
        self.handled_errors: list[Exception] = []

    async def on_error(
        self, error: Exception, deps: None, ctx: ScopedContext | None = None
    ) -> Exception | None:
        self.handled_errors.append(error)
        if self.convert_to:
            return self.convert_to(f"Converted: {error}")
        return None


class TestMiddlewareAgentProperties:
    """Tests for MiddlewareAgent properties."""

    def test_wrapped_property(self) -> None:
        """Test wrapped agent property."""
        agent = Agent(TestModel(), output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.wrapped is agent

    def test_middleware_property(self) -> None:
        """Test middleware list property."""
        agent = Agent(TestModel(), output_type=str)
        mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[mw])
        assert middleware_agent.middleware == [mw]

    def test_context_property(self) -> None:
        """Test context property."""
        agent = Agent(TestModel(), output_type=str)
        ctx = MiddlewareContext(config={"max_retries": 3, "timeout": 30.0})
        middleware_agent = MiddlewareAgent(agent, context=ctx)
        assert middleware_agent.context is ctx

    def test_context_default_none(self) -> None:
        """Test context defaults to None."""
        agent = Agent(TestModel(), output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.context is None

    def test_model_property(self) -> None:
        """Test model property delegates to wrapped."""
        model = TestModel()
        agent = Agent(model, output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.model is model

    def test_name_property(self) -> None:
        """Test name property delegates to wrapped."""
        agent = Agent(TestModel(), output_type=str, name="test_agent")
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.name == "test_agent"

        # Test setter
        middleware_agent.name = "new_name"
        assert middleware_agent.name == "new_name"
        assert agent.name == "new_name"

    def test_deps_type_property(self) -> None:
        """Test deps_type property delegates to wrapped."""
        agent = Agent(TestModel(), deps_type=dict, output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.deps_type is dict

    def test_output_type_property(self) -> None:
        """Test output_type property delegates to wrapped."""
        agent = Agent(TestModel(), output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.output_type is str

    def test_toolsets_property(self) -> None:
        """Test toolsets property delegates to wrapped."""
        agent = Agent(TestModel(), output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.toolsets == agent.toolsets

    def test_event_stream_handler_property(self) -> None:
        """Test event_stream_handler property delegates to wrapped."""
        agent = Agent(TestModel(), output_type=str)
        middleware_agent = MiddlewareAgent(agent)
        assert middleware_agent.event_stream_handler == agent.event_stream_handler


class TestMiddlewareAgentRun:
    """Tests for MiddlewareAgent.run()."""

    async def test_run_with_logging_middleware(self) -> None:
        """Test run with logging middleware."""
        model = TestModel()
        model.custom_output_text = "Hello!"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        await middleware_agent.run("test prompt")

        assert logging_mw.before_run_calls == ["test prompt"]
        assert len(logging_mw.after_run_calls) == 1
        assert logging_mw.after_run_calls[0][0] == "test prompt"

    async def test_run_with_modifying_middleware(self) -> None:
        """Test run with middleware that modifies output."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        modifying_mw = ModifyingMiddleware(prefix="PREFIX")
        middleware_agent = MiddlewareAgent(agent, middleware=[modifying_mw])

        result = await middleware_agent.run("test")

        assert result.output == "PREFIX output: response"

    async def test_run_with_blocking_middleware(self) -> None:
        """Test run with middleware that blocks input."""
        model = TestModel()
        agent = Agent(model, output_type=str)
        blocking_mw = BlockingMiddleware(blocked_words={"forbidden"})
        middleware_agent = MiddlewareAgent(agent, middleware=[blocking_mw])

        with pytest.raises(InputBlocked) as exc_info:
            await middleware_agent.run("This is forbidden content")

        assert "forbidden" in exc_info.value.reason

    async def test_run_with_multiple_middleware(self) -> None:
        """Test run with multiple middleware in order."""
        model = TestModel()
        model.custom_output_text = "base"

        agent = Agent(model, output_type=str)
        mw1 = ModifyingMiddleware(prefix="first")
        mw2 = ModifyingMiddleware(prefix="second")

        middleware_agent = MiddlewareAgent(agent, middleware=[mw1, mw2])
        result = await middleware_agent.run("input")

        # after_run is called in reverse order
        assert "first output:" in result.output
        assert "second output:" in result.output

    async def test_run_with_message_history(self) -> None:
        """Test run with message history (continued conversations)."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        model = TestModel()
        model.custom_output_text = "continued"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        # Create message history for continued conversation
        history = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]

        # This should work with message history
        result = await middleware_agent.run("Continue", message_history=history)
        assert result.output == "continued"

    async def test_run_with_none_prompt_and_instructions(self) -> None:
        """Test run with None prompt but with instructions."""
        model = TestModel()
        model.custom_output_text = "with instructions"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        # Run with None prompt but with instructions
        result = await middleware_agent.run(None, instructions="Do something")
        assert result.output == "with instructions"
        # before_run should not be called with None prompt
        assert logging_mw.before_run_calls == []

    async def test_run_with_additional_toolsets(self) -> None:
        """Test run with additional toolsets wraps them with middleware."""
        from pydantic_ai.toolsets import FunctionToolset

        model = TestModel()
        model.custom_output_text = "done"

        # Create a real toolset with a simple tool
        toolset: FunctionToolset[None] = FunctionToolset()

        @toolset.tool
        def dummy_tool() -> str:
            """A dummy tool."""
            return "result"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        # Run with additional toolsets
        result = await middleware_agent.run("test", toolsets=[toolset])
        assert result.output is not None


class TestMiddlewareAgentErrorHandling:
    """Tests for MiddlewareAgent error handling."""

    async def test_on_error_logs_error(self) -> None:
        """Test that on_error is called on errors."""
        model = TestModel()

        agent = Agent(model, output_type=str)
        error_mw = ErrorHandlingMiddleware()
        blocking_mw = BlockingMiddleware(blocked_words={"block"})

        middleware_agent = MiddlewareAgent(agent, middleware=[error_mw, blocking_mw])

        with pytest.raises(InputBlocked):
            await middleware_agent.run("please block this")

        assert len(error_mw.handled_errors) == 1
        assert isinstance(error_mw.handled_errors[0], InputBlocked)

    async def test_on_error_can_convert_exception(self) -> None:
        """Test that on_error can convert exceptions."""
        model = TestModel()

        agent = Agent(model, output_type=str)
        error_mw = ErrorHandlingMiddleware(convert_to=RuntimeError)
        blocking_mw = BlockingMiddleware(blocked_words={"block"})

        middleware_agent = MiddlewareAgent(agent, middleware=[error_mw, blocking_mw])

        with pytest.raises(RuntimeError) as exc_info:
            await middleware_agent.run("please block this")

        assert "Converted:" in str(exc_info.value)


class TestMiddlewareAgentIter:
    """Tests for MiddlewareAgent.iter()."""

    async def test_iter_with_middleware(self) -> None:
        """Test iter with middleware."""
        model = TestModel()
        model.custom_output_text = "iterated"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        async with middleware_agent.iter("test prompt") as run:
            async for _ in run:
                pass

        assert logging_mw.before_run_calls == ["test prompt"]

    async def test_iter_with_additional_toolsets(self) -> None:
        """Test iter with additional toolsets wraps them with middleware."""
        from pydantic_ai.toolsets import FunctionToolset

        model = TestModel()
        model.custom_output_text = "done"

        # Create a real toolset with a simple tool
        toolset: FunctionToolset[None] = FunctionToolset()

        @toolset.tool
        def dummy_tool() -> str:
            """A dummy tool."""
            return "result"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        # Run iter with additional toolsets
        async with middleware_agent.iter("test", toolsets=[toolset]) as run:
            async for _ in run:
                pass

        assert logging_mw.before_run_calls == ["test"]

    async def test_iter_with_none_prompt_and_instructions(self) -> None:
        """Test iter with None prompt but with instructions."""
        model = TestModel()
        model.custom_output_text = "with instructions"

        agent = Agent(model, output_type=str)
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw])

        # Run iter with None prompt but with instructions
        async with middleware_agent.iter(None, instructions="Do something") as run:
            async for _ in run:
                pass

        # before_run should not be called with None prompt
        assert logging_mw.before_run_calls == []


class TestMiddlewareAgentOverride:
    """Tests for MiddlewareAgent.override()."""

    def test_override_context_manager(self) -> None:
        """Test override context manager."""
        model = TestModel()
        agent = Agent(model, output_type=str, name="original")
        middleware_agent = MiddlewareAgent(agent)

        with middleware_agent.override(name="overridden"):
            assert middleware_agent.name == "overridden"

        assert middleware_agent.name == "original"


class TestMiddlewareAgentWithContext:
    """Tests for MiddlewareAgent with context sharing."""

    async def test_run_with_context_sets_metadata(self) -> None:
        """Test that run with context sets metadata."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        ctx = MiddlewareContext(config={"test_key": "test_value"})
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw], context=ctx)

        result = await middleware_agent.run("test prompt")

        assert result.output == "response"
        # Verify metadata was set
        assert ctx.metadata["user_prompt"] == "test prompt"
        assert "transformed_prompt" in ctx.metadata
        assert "final_output" in ctx.metadata

    async def test_run_with_context_sets_run_usage(self) -> None:
        """Test that run with context sets run_usage in metadata."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        ctx = MiddlewareContext()
        middleware_agent = MiddlewareAgent(agent, middleware=[], context=ctx)

        await middleware_agent.run("test prompt")

        # run_usage should be set in metadata
        assert "run_usage" in ctx.metadata
        run_usage = ctx.metadata["run_usage"]
        # RunUsage has input_tokens and output_tokens
        assert hasattr(run_usage, "input_tokens")
        assert hasattr(run_usage, "output_tokens")

    async def test_run_with_context_provides_config(self) -> None:
        """Test that middleware can access config via context."""

        class ConfigAwareMiddleware(AgentMiddleware[None]):
            def __init__(self) -> None:
                self.config_value: Any = None

            async def before_run(
                self,
                prompt: str | Sequence[Any],
                deps: None,
                ctx: ScopedContext | None = None,
            ) -> str | Sequence[Any]:
                if ctx:
                    self.config_value = ctx.config.get("my_setting")
                return prompt

        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        ctx = MiddlewareContext(config={"my_setting": 42})
        config_mw = ConfigAwareMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[config_mw], context=ctx)

        await middleware_agent.run("test")

        assert config_mw.config_value == 42

    async def test_iter_with_context_sets_metadata(self) -> None:
        """Test that iter with context sets metadata."""
        model = TestModel()
        model.custom_output_text = "response"

        agent = Agent(model, output_type=str)
        ctx = MiddlewareContext(config={"iter_test": True})
        logging_mw = LoggingMiddleware()
        middleware_agent = MiddlewareAgent(agent, middleware=[logging_mw], context=ctx)

        async with middleware_agent.iter("test prompt"):
            pass

        # Verify metadata was set
        assert ctx.metadata["user_prompt"] == "test prompt"
        assert "transformed_prompt" in ctx.metadata


class TestMiddlewareAgentAsyncContext:
    """Tests for MiddlewareAgent async context manager."""

    async def test_aenter_aexit(self) -> None:
        """Test async context manager."""
        model = TestModel()
        agent = Agent(model, output_type=str)
        middleware_agent = MiddlewareAgent(agent)

        async with middleware_agent as ma:
            # Should return the wrapped agent from __aenter__
            assert ma is not None

"""Tests for AsyncGuardrailMiddleware."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from pydantic_ai_middleware import (
    AgentMiddleware,
    AsyncGuardrailMiddleware,
    GuardrailTimeout,
    GuardrailTiming,
    InputBlocked,
    ScopedContext,
)


class SlowGuardrail(AgentMiddleware[None]):
    """Guardrail that takes time to execute."""

    def __init__(self, delay: float, should_fail: bool = False) -> None:
        self.delay = delay
        self.should_fail = should_fail
        self.before_run_called = False
        self.after_run_called = False
        self.before_tool_called = False
        self.after_tool_called = False
        self.before_model_called = False

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        await asyncio.sleep(self.delay)
        self.before_run_called = True
        if self.should_fail:
            raise InputBlocked("Guardrail blocked")
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        await asyncio.sleep(self.delay)
        self.after_run_called = True
        if self.should_fail:
            raise InputBlocked("Guardrail blocked output")
        return output

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        self.before_tool_called = True
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        self.after_tool_called = True
        return result

    async def before_model_request(
        self, messages: list, deps: None, ctx: ScopedContext | None = None
    ) -> list:
        self.before_model_called = True
        return messages

    async def on_error(
        self, error: Exception, deps: None, ctx: ScopedContext | None = None
    ) -> Exception | None:
        return ValueError(f"Converted: {error}")


class PassThroughGuardrail(AgentMiddleware[None]):
    """Simple guardrail that passes through everything."""

    def __init__(self) -> None:
        self.before_run_count = 0
        self.after_run_count = 0

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        self.before_run_count += 1
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        self.after_run_count += 1
        return output


class TestAsyncGuardrailInit:
    """Tests for AsyncGuardrailMiddleware initialization."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail)

        assert async_mw.guardrail is guardrail
        assert async_mw.timing == GuardrailTiming.CONCURRENT
        assert async_mw.cancel_on_failure is True
        assert async_mw.timeout is None

    def test_init_with_blocking_timing(self) -> None:
        """Test initialization with blocking timing."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.BLOCKING)

        assert async_mw.timing == GuardrailTiming.BLOCKING

    def test_init_with_async_post_timing(self) -> None:
        """Test initialization with async_post timing."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.ASYNC_POST)

        assert async_mw.timing == GuardrailTiming.ASYNC_POST

    def test_init_with_timeout(self) -> None:
        """Test initialization with timeout."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timeout=5.0)

        assert async_mw.timeout == 5.0

    def test_init_with_name(self) -> None:
        """Test initialization with custom name."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, name="my_guardrail")

        assert async_mw.name == "my_guardrail"

    def test_init_cancel_on_failure_false(self) -> None:
        """Test initialization with cancel_on_failure=False."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, cancel_on_failure=False)

        assert async_mw.cancel_on_failure is False

    def test_repr(self) -> None:
        """Test string representation."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail)

        repr_str = repr(async_mw)
        assert "AsyncGuardrailMiddleware" in repr_str
        assert "PassThroughGuardrail" in repr_str
        assert "concurrent" in repr_str


class TestAsyncGuardrailBlockingMode:
    """Tests for BLOCKING timing mode."""

    async def test_blocking_mode_waits(self) -> None:
        """Test that blocking mode waits for guardrail to complete."""
        guardrail = SlowGuardrail(delay=0.1)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.BLOCKING)

        start = asyncio.get_event_loop().time()
        result = await async_mw.before_run("test", None)
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.1
        assert guardrail.before_run_called
        assert result == "test"

    async def test_blocking_mode_raises_on_failure(self) -> None:
        """Test that blocking mode raises when guardrail fails."""
        guardrail = SlowGuardrail(delay=0.01, should_fail=True)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.BLOCKING)

        with pytest.raises(InputBlocked, match="Guardrail blocked"):
            await async_mw.before_run("test", None)

    async def test_blocking_mode_timeout(self) -> None:
        """Test that blocking mode respects timeout."""
        guardrail = SlowGuardrail(delay=1.0)
        async_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail, timing=GuardrailTiming.BLOCKING, timeout=0.1
        )

        with pytest.raises(GuardrailTimeout) as exc_info:
            await async_mw.before_run("test", None)

        assert exc_info.value.timeout == 0.1

    async def test_blocking_mode_passes_tool_calls(self) -> None:
        """Test that blocking mode passes through tool call hooks."""
        guardrail = SlowGuardrail(delay=0.01)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.BLOCKING)

        await async_mw.before_tool_call("tool", {}, None)
        await async_mw.after_tool_call("tool", {}, "result", None)

        assert guardrail.before_tool_called
        assert guardrail.after_tool_called

    async def test_blocking_mode_passes_model_request(self) -> None:
        """Test that blocking mode passes through model request hook."""
        guardrail = SlowGuardrail(delay=0.01)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.BLOCKING)

        await async_mw.before_model_request([], None)

        assert guardrail.before_model_called


class TestAsyncGuardrailConcurrentMode:
    """Tests for CONCURRENT timing mode."""

    async def test_concurrent_mode_returns_immediately(self) -> None:
        """Test that concurrent mode returns immediately."""
        guardrail = SlowGuardrail(delay=0.5)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        start = asyncio.get_event_loop().time()
        result = await async_mw.before_run("test", None)
        elapsed = asyncio.get_event_loop().time() - start

        # Should return almost immediately (guardrail runs in background)
        assert elapsed < 0.1
        assert result == "test"

    async def test_concurrent_mode_checks_on_after_run(self) -> None:
        """Test that concurrent mode checks guardrail result in after_run."""
        guardrail = SlowGuardrail(delay=0.1)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        await async_mw.before_run("test", None)
        # Wait for after_run which waits for the guardrail
        await async_mw.after_run("test", "output", None)

        assert guardrail.before_run_called

    async def test_concurrent_mode_failure_blocks_output(self) -> None:
        """Test that concurrent mode blocks output when guardrail fails."""
        guardrail = SlowGuardrail(delay=0.05, should_fail=True)
        async_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail, timing=GuardrailTiming.CONCURRENT, cancel_on_failure=True
        )

        await async_mw.before_run("test", None)

        with pytest.raises(InputBlocked, match="Guardrail failed"):
            await async_mw.after_run("test", "output", None)

    async def test_concurrent_mode_failure_no_block_when_disabled(self) -> None:
        """Test that concurrent mode doesn't block when cancel_on_failure=False."""
        guardrail = SlowGuardrail(delay=0.05, should_fail=True)
        async_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail,
            timing=GuardrailTiming.CONCURRENT,
            cancel_on_failure=False,
        )

        await async_mw.before_run("test", None)
        # Should not raise, just log
        result = await async_mw.after_run("test", "output", None)

        assert result == "output"

    async def test_concurrent_mode_timeout_in_after_run(self) -> None:
        """Test that concurrent mode times out in after_run."""
        guardrail = SlowGuardrail(delay=1.0)
        async_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail, timing=GuardrailTiming.CONCURRENT, timeout=0.1
        )

        await async_mw.before_run("test", None)

        with pytest.raises(GuardrailTimeout) as exc_info:
            await async_mw.after_run("test", "output", None)

        assert exc_info.value.timeout == 0.1

    async def test_concurrent_mode_skips_tool_calls(self) -> None:
        """Test that concurrent mode skips tool call hooks."""
        guardrail = SlowGuardrail(delay=0.01)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        result = await async_mw.before_tool_call("tool", {"arg": 1}, None)
        assert result == {"arg": 1}
        assert not guardrail.before_tool_called

    async def test_concurrent_mode_skips_model_request(self) -> None:
        """Test that concurrent mode skips model request hook."""
        guardrail = SlowGuardrail(delay=0.01)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        result = await async_mw.before_model_request([], None)
        assert result == []
        assert not guardrail.before_model_called

    async def test_concurrent_mode_skips_after_tool_call(self) -> None:
        """Test that concurrent mode skips after_tool_call hook."""
        guardrail = SlowGuardrail(delay=0.01)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        result = await async_mw.after_tool_call("tool", {"arg": 1}, "result", None)
        assert result == "result"
        assert not guardrail.after_tool_called


class TestAsyncGuardrailAsyncPostMode:
    """Tests for ASYNC_POST timing mode."""

    async def test_async_post_mode_skips_before_run(self) -> None:
        """Test that async_post mode skips before_run."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.ASYNC_POST)

        result = await async_mw.before_run("test", None)

        assert result == "test"
        assert guardrail.before_run_count == 0

    async def test_async_post_mode_fires_in_after_run(self) -> None:
        """Test that async_post mode launches guardrail in after_run."""
        guardrail = SlowGuardrail(delay=0.1)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.ASYNC_POST)

        await async_mw.before_run("test", None)
        result = await async_mw.after_run("test", "output", None)

        # Returns immediately, guardrail runs in background
        assert result == "output"

        # Wait a bit for background task
        await asyncio.sleep(0.2)
        assert guardrail.after_run_called

    async def test_async_post_mode_doesnt_block_on_failure(self) -> None:
        """Test that async_post mode doesn't block on guardrail failure."""
        guardrail = SlowGuardrail(delay=0.05, should_fail=True)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.ASYNC_POST)

        await async_mw.before_run("test", None)
        # Should not raise even though guardrail will fail
        result = await async_mw.after_run("test", "output", None)

        assert result == "output"


class TestAsyncGuardrailOnError:
    """Tests for AsyncGuardrailMiddleware.on_error()."""

    async def test_on_error_delegates_to_guardrail(self) -> None:
        """Test that on_error delegates to wrapped guardrail."""
        guardrail = SlowGuardrail(delay=0.01)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail)

        error = ValueError("original")
        result = await async_mw.on_error(error, None)

        assert result is not None
        assert "Converted" in str(result)


class TestAsyncGuardrailStateReset:
    """Tests for state reset between runs."""

    async def test_state_resets_between_runs(self) -> None:
        """Test that internal state resets between runs."""
        guardrail = SlowGuardrail(delay=0.05)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        # First run
        await async_mw.before_run("test1", None)
        await async_mw.after_run("test1", "output1", None)

        # Reset the guardrail's state
        guardrail.before_run_called = False

        # Second run
        await async_mw.before_run("test2", None)
        await async_mw.after_run("test2", "output2", None)

        assert guardrail.before_run_called


class TestAsyncGuardrailIntegration:
    """Integration tests for AsyncGuardrailMiddleware."""

    async def test_concurrent_saves_time_on_success(self) -> None:
        """Test that concurrent mode saves time when guardrail passes."""
        guardrail = SlowGuardrail(delay=0.2)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        async def simulate_llm_call():
            await asyncio.sleep(0.3)
            return "LLM output"

        start = asyncio.get_event_loop().time()

        # Start guardrail in background
        await async_mw.before_run("test", None)

        # Simulate LLM call (runs concurrently with guardrail)
        llm_output = await simulate_llm_call()

        # Check guardrail result
        result = await async_mw.after_run("test", llm_output, None)

        elapsed = asyncio.get_event_loop().time() - start

        # Total time should be ~0.3s (LLM time), not 0.5s (guardrail + LLM)
        assert elapsed < 0.4
        assert result == "LLM output"

    async def test_concurrent_saves_time_on_failure(self) -> None:
        """Test that concurrent mode saves time when guardrail fails early."""
        # Guardrail fails fast (0.1s), LLM would take longer
        guardrail = SlowGuardrail(delay=0.1, should_fail=True)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        start = asyncio.get_event_loop().time()

        await async_mw.before_run("test", None)

        # Wait a bit (in real scenario, LLM would be running)
        await asyncio.sleep(0.15)

        with pytest.raises(InputBlocked):
            await async_mw.after_run("test", "would be discarded", None)

        elapsed = asyncio.get_event_loop().time() - start

        # Total time is ~0.15s (our simulated LLM time)
        # In real scenario, we'd cancel the LLM at this point
        assert elapsed < 0.25

    async def test_blocking_vs_concurrent_timing_comparison(self) -> None:
        """Compare timing between blocking and concurrent modes."""
        guardrail_delay = 0.1
        llm_delay = 0.15

        async def simulate_llm():
            await asyncio.sleep(llm_delay)
            return "output"

        # Blocking mode
        guardrail1 = SlowGuardrail(delay=guardrail_delay)
        blocking_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail1, timing=GuardrailTiming.BLOCKING
        )

        start = asyncio.get_event_loop().time()
        await blocking_mw.before_run("test", None)
        await simulate_llm()
        blocking_time = asyncio.get_event_loop().time() - start

        # Concurrent mode
        guardrail2 = SlowGuardrail(delay=guardrail_delay)
        concurrent_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail2, timing=GuardrailTiming.CONCURRENT
        )

        start = asyncio.get_event_loop().time()
        await concurrent_mw.before_run("test", None)
        llm_output = await simulate_llm()
        await concurrent_mw.after_run("test", llm_output, None)
        concurrent_time = asyncio.get_event_loop().time() - start

        # Concurrent should be faster
        # Blocking: 0.1 + 0.15 = 0.25s
        # Concurrent: max(0.1, 0.15) = 0.15s
        assert concurrent_time < blocking_time

    async def test_async_post_with_timeout(self) -> None:
        """Test async_post mode with timeout for monitoring guardrail."""
        guardrail = SlowGuardrail(delay=1.0)  # Slow guardrail
        async_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail,
            timing=GuardrailTiming.ASYNC_POST,
            timeout=0.1,  # Should timeout
        )

        await async_mw.before_run("test", None)
        # This should not raise even though monitoring guardrail will timeout
        result = await async_mw.after_run("test", "output", None)

        assert result == "output"
        # Wait for background task to complete/timeout
        await asyncio.sleep(0.2)

    async def test_async_post_with_guardrail_error(self) -> None:
        """Test async_post mode when guardrail raises an error."""
        guardrail = SlowGuardrail(delay=0.05, should_fail=True)
        async_mw = AsyncGuardrailMiddleware(
            guardrail=guardrail,
            timing=GuardrailTiming.ASYNC_POST,
        )

        await async_mw.before_run("test", None)
        # This should not raise - async_post is fire-and-forget
        result = await async_mw.after_run("test", "output", None)

        assert result == "output"
        # Wait for background task
        await asyncio.sleep(0.1)

    async def test_concurrent_after_run_without_task(self) -> None:
        """Test concurrent after_run when no task was started (edge case)."""
        guardrail = PassThroughGuardrail()
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        # Skip before_run, call after_run directly
        # This tests the case where _guardrail_task is None
        result = await async_mw.after_run("test", "output", None)
        assert result == "output"

    async def test_concurrent_with_cancelled_task(self) -> None:
        """Test concurrent mode when guardrail task is cancelled externally."""
        guardrail = SlowGuardrail(delay=1.0)
        async_mw = AsyncGuardrailMiddleware(guardrail=guardrail, timing=GuardrailTiming.CONCURRENT)

        await async_mw.before_run("test", None)

        # Cancel the task externally
        if async_mw._guardrail_task:
            async_mw._guardrail_task.cancel()

        # Should handle CancelledError gracefully
        result = await async_mw.after_run("test", "output", None)
        assert result == "output"

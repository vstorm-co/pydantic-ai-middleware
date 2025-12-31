"""Tests for ParallelMiddleware."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage

from pydantic_ai_middleware import (
    AgentMiddleware,
    AggregationFailed,
    AggregationStrategy,
    InputBlocked,
    ParallelMiddleware,
)


class SlowMiddleware(AgentMiddleware[None]):
    """Middleware that takes time to execute."""

    def __init__(self, delay: float, name: str = "slow") -> None:
        self.delay = delay
        self.name = name
        self.before_run_called = False
        self.after_run_called = False

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None
    ) -> str | Sequence[Any]:
        await asyncio.sleep(self.delay)
        self.before_run_called = True
        return f"{self.name}:{prompt}"

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None
    ) -> Any:
        await asyncio.sleep(self.delay)
        self.after_run_called = True
        return f"{self.name}:{output}"


class FailingMiddleware(AgentMiddleware[None]):
    """Middleware that always raises an exception."""

    def __init__(self, error_message: str = "Test error") -> None:
        self.error_message = error_message

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None
    ) -> str | Sequence[Any]:
        raise InputBlocked(self.error_message)

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None
    ) -> Any:
        raise InputBlocked(self.error_message)


class SlowFailingMiddleware(AgentMiddleware[None]):
    """Middleware that waits then raises an exception."""

    def __init__(self, delay: float, reason: str = "Test error") -> None:
        self.delay = delay
        self.reason = reason

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None
    ) -> str | Sequence[Any]:
        await asyncio.sleep(self.delay)
        raise InputBlocked(self.reason)

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None
    ) -> Any:
        await asyncio.sleep(self.delay)
        raise InputBlocked(self.reason)


class CountingMiddleware(AgentMiddleware[None]):
    """Middleware that counts how many times it was called."""

    def __init__(self) -> None:
        self.before_run_count = 0
        self.after_run_count = 0
        self.before_tool_count = 0
        self.after_tool_count = 0
        self.before_model_count = 0
        self.on_error_count = 0

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None
    ) -> str | Sequence[Any]:
        self.before_run_count += 1
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None
    ) -> Any:
        self.after_run_count += 1
        return output

    async def before_tool_call(
        self, tool_name: str, tool_args: dict[str, Any], deps: None
    ) -> dict[str, Any]:
        self.before_tool_count += 1
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
    ) -> Any:
        self.after_tool_count += 1
        return result

    async def before_model_request(
        self, messages: list[ModelMessage], deps: None
    ) -> list[ModelMessage]:
        self.before_model_count += 1
        return messages

    async def on_error(self, error: Exception, deps: None) -> Exception | None:
        self.on_error_count += 1
        return None


class ErrorHandlingMiddleware(AgentMiddleware[None]):
    """Middleware that can handle errors."""

    def __init__(self, should_handle: bool = True) -> None:
        self.should_handle = should_handle
        self.handled_error: Exception | None = None

    async def on_error(self, error: Exception, deps: None) -> Exception | None:
        self.handled_error = error
        if self.should_handle:
            return ValueError(f"Converted: {error}")
        return None


class TestParallelMiddlewareInit:
    """Tests for ParallelMiddleware initialization."""

    def test_init_with_single_middleware(self) -> None:
        """Test initialization with a single middleware."""
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw])
        assert len(parallel.middleware) == 1
        assert parallel.strategy == AggregationStrategy.ALL_MUST_PASS

    def test_init_with_multiple_middleware(self) -> None:
        """Test initialization with multiple middleware."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        mw3 = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw1, mw2, mw3])
        assert len(parallel.middleware) == 3

    def test_init_with_custom_strategy(self) -> None:
        """Test initialization with custom aggregation strategy."""
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(
            middleware=[mw], strategy=AggregationStrategy.FIRST_SUCCESS
        )
        assert parallel.strategy == AggregationStrategy.FIRST_SUCCESS

    def test_init_with_timeout(self) -> None:
        """Test initialization with timeout."""
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw], timeout=5.0)
        assert parallel.timeout == 5.0

    def test_init_with_name(self) -> None:
        """Test initialization with custom name."""
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw], name="my_parallel")
        assert parallel.name == "my_parallel"

    def test_init_empty_middleware_raises(self) -> None:
        """Test that empty middleware list raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one middleware"):
            ParallelMiddleware(middleware=[])

    def test_repr(self) -> None:
        """Test string representation."""
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw], timeout=5.0)
        repr_str = repr(parallel)
        assert "ParallelMiddleware" in repr_str
        assert "strategy=all_must_pass" in repr_str
        assert "timeout=5.0" in repr_str


class TestParallelMiddlewareBeforeRun:
    """Tests for ParallelMiddleware.before_run()."""

    async def test_before_run_executes_all(self) -> None:
        """Test that before_run executes all middleware."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        mw3 = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw1, mw2, mw3])

        await parallel.before_run("test", None)

        assert mw1.before_run_count == 1
        assert mw2.before_run_count == 1
        assert mw3.before_run_count == 1

    async def test_before_run_parallel_execution(self) -> None:
        """Test that before_run executes middleware in parallel."""
        # Each middleware takes 0.1s
        mw1 = SlowMiddleware(0.1, "mw1")
        mw2 = SlowMiddleware(0.1, "mw2")
        mw3 = SlowMiddleware(0.1, "mw3")
        parallel = ParallelMiddleware(middleware=[mw1, mw2, mw3])

        start = asyncio.get_event_loop().time()
        await parallel.before_run("test", None)
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in ~0.1s, not ~0.3s (sequential)
        assert elapsed < 0.2  # Allow some margin
        assert mw1.before_run_called
        assert mw2.before_run_called
        assert mw3.before_run_called

    async def test_before_run_all_must_pass_success(self) -> None:
        """Test ALL_MUST_PASS strategy with all successes."""
        mw1 = SlowMiddleware(0.01, "mw1")
        mw2 = SlowMiddleware(0.01, "mw2")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.ALL_MUST_PASS
        )

        result = await parallel.before_run("test", None)

        # Returns one of the successful results (order not guaranteed in parallel)
        assert result in ("mw1:test", "mw2:test")

    async def test_before_run_all_must_pass_failure(self) -> None:
        """Test ALL_MUST_PASS strategy with one failure."""
        mw1 = CountingMiddleware()
        mw2 = FailingMiddleware("blocked")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.ALL_MUST_PASS
        )

        with pytest.raises(InputBlocked, match="blocked"):
            await parallel.before_run("test", None)

    async def test_before_run_first_success_returns_first(self) -> None:
        """Test FIRST_SUCCESS returns first successful result."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.FIRST_SUCCESS
        )

        result = await parallel.before_run("test", None)

        assert result == "test"

    async def test_before_run_first_success_all_fail(self) -> None:
        """Test FIRST_SUCCESS raises when all fail."""
        mw1 = FailingMiddleware("error1")
        mw2 = FailingMiddleware("error2")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.FIRST_SUCCESS
        )

        with pytest.raises(AggregationFailed) as exc_info:
            await parallel.before_run("test", None)

        assert exc_info.value.strategy == AggregationStrategy.FIRST_SUCCESS
        assert len(exc_info.value.errors) == 2

    async def test_before_run_race_returns_first_completed(self) -> None:
        """Test RACE returns first completed result."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.RACE
        )

        result = await parallel.before_run("test", None)

        assert result == "test"

    async def test_before_run_collect_all(self) -> None:
        """Test COLLECT_ALL returns all results."""
        mw1 = SlowMiddleware(0.01, "mw1")
        mw2 = SlowMiddleware(0.01, "mw2")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.COLLECT_ALL
        )

        result = await parallel.before_run("test", None)

        assert isinstance(result, list)
        assert len(result) == 2
        # Each is a (success, value) tuple
        assert result[0] == (True, "mw1:test")
        assert result[1] == (True, "mw2:test")

    async def test_before_run_timeout(self) -> None:
        """Test timeout raises asyncio.TimeoutError."""
        mw = SlowMiddleware(1.0, "slow")  # 1 second delay
        parallel = ParallelMiddleware(middleware=[mw], timeout=0.1)

        with pytest.raises(asyncio.TimeoutError):
            await parallel.before_run("test", None)

    async def test_before_run_collect_all_with_timeout(self) -> None:
        """Test COLLECT_ALL with timeout uses _execute_all path."""
        mw1 = SlowMiddleware(0.01, "mw1")
        mw2 = SlowMiddleware(0.01, "mw2")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2],
            strategy=AggregationStrategy.COLLECT_ALL,
            timeout=5.0,
        )

        result = await parallel.before_run("test", None)

        assert isinstance(result, list)
        assert len(result) == 2


class TestParallelMiddlewareAfterRun:
    """Tests for ParallelMiddleware.after_run()."""

    async def test_after_run_executes_all(self) -> None:
        """Test that after_run executes all middleware."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        await parallel.after_run("prompt", "output", None)

        assert mw1.after_run_count == 1
        assert mw2.after_run_count == 1

    async def test_after_run_parallel_execution(self) -> None:
        """Test that after_run executes in parallel."""
        mw1 = SlowMiddleware(0.1, "mw1")
        mw2 = SlowMiddleware(0.1, "mw2")
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        start = asyncio.get_event_loop().time()
        await parallel.after_run("prompt", "output", None)
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in ~0.1s, not ~0.2s
        assert elapsed < 0.15
        assert mw1.after_run_called
        assert mw2.after_run_called


class TestParallelMiddlewareToolCalls:
    """Tests for ParallelMiddleware tool call hooks."""

    async def test_before_tool_call(self) -> None:
        """Test before_tool_call executes all middleware."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        await parallel.before_tool_call("test_tool", {"arg": "value"}, None)

        assert mw1.before_tool_count == 1
        assert mw2.before_tool_count == 1

    async def test_after_tool_call(self) -> None:
        """Test after_tool_call executes all middleware."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        await parallel.after_tool_call("test_tool", {"arg": "value"}, "result", None)

        assert mw1.after_tool_count == 1
        assert mw2.after_tool_count == 1


class TestParallelMiddlewareModelRequest:
    """Tests for ParallelMiddleware.before_model_request()."""

    async def test_before_model_request(self) -> None:
        """Test before_model_request executes all middleware."""
        mw1 = CountingMiddleware()
        mw2 = CountingMiddleware()
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        await parallel.before_model_request([], None)

        assert mw1.before_model_count == 1
        assert mw2.before_model_count == 1


class TestParallelMiddlewareOnError:
    """Tests for ParallelMiddleware.on_error()."""

    async def test_on_error_returns_first_handler(self) -> None:
        """Test on_error returns first non-None result."""
        mw1 = ErrorHandlingMiddleware(should_handle=True)
        mw2 = ErrorHandlingMiddleware(should_handle=True)
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        error = ValueError("original")
        result = await parallel.on_error(error, None)

        assert result is not None
        assert "Converted" in str(result)

    async def test_on_error_returns_none_if_all_none(self) -> None:
        """Test on_error returns None if all handlers return None."""
        mw1 = ErrorHandlingMiddleware(should_handle=False)
        mw2 = ErrorHandlingMiddleware(should_handle=False)
        parallel = ParallelMiddleware(middleware=[mw1, mw2])

        error = ValueError("original")
        result = await parallel.on_error(error, None)

        assert result is None


class TestParallelMiddlewareIntegration:
    """Integration tests for ParallelMiddleware."""

    async def test_mixed_success_and_failure_first_success(self) -> None:
        """Test mixed results with FIRST_SUCCESS strategy."""
        mw1 = FailingMiddleware("error1")
        mw2 = CountingMiddleware()  # Will succeed
        mw3 = FailingMiddleware("error3")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2, mw3], strategy=AggregationStrategy.FIRST_SUCCESS
        )

        # Should not raise, as one middleware succeeded
        result = await parallel.before_run("test", None)
        assert result == "test"

    async def test_nested_parallel_middleware(self) -> None:
        """Test ParallelMiddleware containing another ParallelMiddleware."""
        inner1 = CountingMiddleware()
        inner2 = CountingMiddleware()
        inner_parallel = ParallelMiddleware(middleware=[inner1, inner2])

        outer1 = CountingMiddleware()
        outer_parallel = ParallelMiddleware(middleware=[inner_parallel, outer1])

        await outer_parallel.before_run("test", None)

        assert inner1.before_run_count == 1
        assert inner2.before_run_count == 1
        assert outer1.before_run_count == 1

    async def test_on_error_with_timeout(self) -> None:
        """Test on_error with timeout parameter set."""
        mw1 = ErrorHandlingMiddleware(should_handle=True)
        parallel = ParallelMiddleware(middleware=[mw1], timeout=5.0)

        error = ValueError("original")
        result = await parallel.on_error(error, None)

        assert result is not None

    async def test_race_with_failure_first(self) -> None:
        """Test RACE strategy where failure completes first."""
        # Use delays to ensure the failing middleware completes first
        mw1 = SlowFailingMiddleware(delay=0.01, reason="fast_error")
        mw2 = SlowMiddleware(delay=0.5, name="slow")
        parallel = ParallelMiddleware(
            middleware=[mw1, mw2], strategy=AggregationStrategy.RACE
        )

        with pytest.raises(InputBlocked, match="fast_error"):
            await parallel.before_run("test", None)

    async def test_race_with_empty_results(self) -> None:
        """Test RACE strategy when results list is logically empty (edge case)."""
        # Create a custom middleware that we can verify runs
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(
            middleware=[mw], strategy=AggregationStrategy.RACE
        )

        result = await parallel.before_run("test", None)
        assert result == "test"

    async def test_first_success_with_no_results(self) -> None:
        """Test FIRST_SUCCESS returns default when all succeed but list empty edge."""
        mw = CountingMiddleware()
        parallel = ParallelMiddleware(
            middleware=[mw], strategy=AggregationStrategy.FIRST_SUCCESS
        )

        result = await parallel.before_run("test", None)
        assert result == "test"


class TestParallelMiddlewareEarlyCancellation:
    """Tests for early cancellation behavior."""

    async def test_all_must_pass_cancels_on_failure(self) -> None:
        """Test ALL_MUST_PASS cancels remaining tasks on first failure."""
        # Fast failure, slow success
        fast_fail = SlowFailingMiddleware(delay=0.01, reason="fast_fail")
        slow_success = SlowMiddleware(delay=1.0, name="slow")

        parallel = ParallelMiddleware(
            middleware=[fast_fail, slow_success],
            strategy=AggregationStrategy.ALL_MUST_PASS,
        )

        import time

        start = time.time()
        with pytest.raises(InputBlocked, match="fast_fail"):
            await parallel.before_run("test", None)
        elapsed = time.time() - start

        # Should complete quickly (not wait for 1s slow middleware)
        assert elapsed < 0.5

    async def test_first_success_cancels_remaining(self) -> None:
        """Test FIRST_SUCCESS cancels remaining tasks on first success."""
        fast_success = SlowMiddleware(delay=0.01, name="fast")
        slow_success = SlowMiddleware(delay=1.0, name="slow")

        parallel = ParallelMiddleware(
            middleware=[fast_success, slow_success],
            strategy=AggregationStrategy.FIRST_SUCCESS,
        )

        import time

        start = time.time()
        result = await parallel.before_run("test", None)
        elapsed = time.time() - start

        # Should complete quickly (not wait for 1s slow middleware)
        assert elapsed < 0.5
        assert result == "fast:test"

    async def test_race_cancels_on_first_completion(self) -> None:
        """Test RACE cancels remaining tasks on first completion."""
        fast = SlowMiddleware(delay=0.01, name="fast")
        slow = SlowMiddleware(delay=1.0, name="slow")

        parallel = ParallelMiddleware(
            middleware=[fast, slow],
            strategy=AggregationStrategy.RACE,
        )

        import time

        start = time.time()
        result = await parallel.before_run("test", None)
        elapsed = time.time() - start

        # Should complete quickly (not wait for 1s slow middleware)
        assert elapsed < 0.5
        assert result == "fast:test"

    async def test_collect_all_waits_for_all(self) -> None:
        """Test COLLECT_ALL waits for all tasks to complete."""
        fast = SlowMiddleware(delay=0.01, name="fast")
        slow = SlowMiddleware(delay=0.2, name="slow")

        parallel = ParallelMiddleware(
            middleware=[fast, slow],
            strategy=AggregationStrategy.COLLECT_ALL,
        )

        import time

        start = time.time()
        result = await parallel.before_run("test", None)
        elapsed = time.time() - start

        # Should wait for slow middleware
        assert elapsed >= 0.2
        # Returns list of all results
        assert len(result) == 2

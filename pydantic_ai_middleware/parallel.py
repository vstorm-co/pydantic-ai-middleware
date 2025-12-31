"""Parallel middleware execution for running multiple middleware concurrently."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Generic

from pydantic_ai.messages import ModelMessage

from .base import AgentMiddleware, DepsT
from .exceptions import AggregationFailed
from .strategies import AggregationStrategy


class ParallelMiddleware(AgentMiddleware[DepsT], Generic[DepsT]):
    """Execute multiple middleware instances in parallel.

    This middleware wraps multiple middleware instances and executes their
    hooks concurrently using asyncio.gather, reducing total blocking time
    when you have multiple independent checks (e.g., validators).

    Example:
        ```python
        from pydantic_ai_middleware import (
            MiddlewareAgent,
            ParallelMiddleware,
            AggregationStrategy,
        )

        # Run 3 validators concurrently instead of sequentially
        parallel_validators = ParallelMiddleware(
            middleware=[
                ProfanityFilter(),
                PIIDetector(),
                PromptInjectionGuard(),
            ],
            strategy=AggregationStrategy.ALL_MUST_PASS,
        )

        agent = MiddlewareAgent(
            agent=base_agent,
            middleware=[parallel_validators],
        )

        # All 3 validators run concurrently
        # If each takes 1s, total time is ~1s instead of ~3s
        result = await agent.run("User input")
        ```

    Performance:
        - Sequential: N middleware × T seconds each = N*T total
        - Parallel: max(T1, T2, ..., TN) ≈ T total (N× speedup)

    Thread Safety:
        All execution is async and uses asyncio.gather for concurrency.
        No threading is involved.
    """

    def __init__(
        self,
        middleware: Sequence[AgentMiddleware[DepsT]],
        strategy: AggregationStrategy = AggregationStrategy.ALL_MUST_PASS,
        timeout: float | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize parallel middleware.

        Args:
            middleware: List of middleware instances to execute in parallel.
                All middleware will receive the same input and run concurrently.
            strategy: How to aggregate results from parallel execution.
                Defaults to ALL_MUST_PASS (all must succeed).
            timeout: Maximum time in seconds to wait for all middleware.
                If exceeded, raises asyncio.TimeoutError.
            name: Optional name for debugging and logging.

        Raises:
            ValueError: If middleware list is empty.
        """
        if not middleware:
            raise ValueError("ParallelMiddleware requires at least one middleware")

        self.middleware = list(middleware)
        self.strategy = strategy
        self.timeout = timeout
        self.name = name or f"ParallelMiddleware[{len(middleware)}]"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ParallelMiddleware("
            f"middleware={len(self.middleware)}, "
            f"strategy={self.strategy.value}, "
            f"timeout={self.timeout})"
        )

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: DepsT | None,
    ) -> str | Sequence[Any]:
        """Execute all middleware before_run hooks in parallel.

        Args:
            prompt: The user prompt to validate/transform.
            deps: The agent dependencies.

        Returns:
            The (possibly modified) prompt based on aggregation strategy.

        Raises:
            ParallelExecutionFailed: If strategy is ALL_MUST_PASS and any fails.
            AggregationFailed: If aggregation cannot produce a valid result.
            asyncio.TimeoutError: If timeout is exceeded.
        """
        tasks = [mw.before_run(prompt, deps) for mw in self.middleware]
        results = await self._execute_parallel(tasks)
        return self._aggregate_results(results, prompt)

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: DepsT | None,
    ) -> Any:
        """Execute all middleware after_run hooks in parallel.

        Args:
            prompt: The original user prompt.
            output: The agent output to validate/transform.
            deps: The agent dependencies.

        Returns:
            The (possibly modified) output based on aggregation strategy.

        Raises:
            ParallelExecutionFailed: If strategy is ALL_MUST_PASS and any fails.
            AggregationFailed: If aggregation cannot produce a valid result.
            asyncio.TimeoutError: If timeout is exceeded.
        """
        tasks = [mw.after_run(prompt, output, deps) for mw in self.middleware]
        results = await self._execute_parallel(tasks)
        return self._aggregate_results(results, output)

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: DepsT | None,
    ) -> list[ModelMessage]:
        """Execute all middleware before_model_request hooks in parallel.

        Args:
            messages: The messages to send to the model.
            deps: The agent dependencies.

        Returns:
            The (possibly modified) messages based on aggregation strategy.

        Raises:
            ParallelExecutionFailed: If strategy is ALL_MUST_PASS and any fails.
            AggregationFailed: If aggregation cannot produce a valid result.
            asyncio.TimeoutError: If timeout is exceeded.
        """
        tasks = [mw.before_model_request(messages, deps) for mw in self.middleware]
        results = await self._execute_parallel(tasks)
        return self._aggregate_results(results, messages)

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: DepsT | None,
    ) -> dict[str, Any]:
        """Execute all middleware before_tool_call hooks in parallel.

        Args:
            tool_name: The name of the tool being called.
            tool_args: The arguments to the tool.
            deps: The agent dependencies.

        Returns:
            The (possibly modified) tool arguments based on aggregation strategy.

        Raises:
            ParallelExecutionFailed: If strategy is ALL_MUST_PASS and any fails.
            AggregationFailed: If aggregation cannot produce a valid result.
            asyncio.TimeoutError: If timeout is exceeded.
        """
        tasks = [mw.before_tool_call(tool_name, tool_args, deps) for mw in self.middleware]
        results = await self._execute_parallel(tasks)
        return self._aggregate_results(results, tool_args)

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: DepsT | None,
    ) -> Any:
        """Execute all middleware after_tool_call hooks in parallel.

        Args:
            tool_name: The name of the tool that was called.
            tool_args: The arguments that were passed to the tool.
            result: The result from the tool.
            deps: The agent dependencies.

        Returns:
            The (possibly modified) result based on aggregation strategy.

        Raises:
            ParallelExecutionFailed: If strategy is ALL_MUST_PASS and any fails.
            AggregationFailed: If aggregation cannot produce a valid result.
            asyncio.TimeoutError: If timeout is exceeded.
        """
        tasks = [mw.after_tool_call(tool_name, tool_args, result, deps) for mw in self.middleware]
        results = await self._execute_parallel(tasks)
        return self._aggregate_results(results, result)

    async def on_error(
        self,
        error: Exception,
        deps: DepsT | None,
    ) -> Exception | None:
        """Execute all middleware on_error hooks in parallel.

        For error handling, we use FIRST_SUCCESS semantics: the first
        middleware that returns a non-None exception wins. If all return
        None, we return None (re-raise original).

        Args:
            error: The exception that occurred.
            deps: The agent dependencies.

        Returns:
            A different exception to raise, or None to re-raise the original.
        """
        tasks = [mw.on_error(error, deps) for mw in self.middleware]

        # For on_error, we don't use return_exceptions because we want
        # the actual return values (Exception | None), not to catch exceptions
        # raised during execution.
        if self.timeout is not None:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.timeout,
            )
        else:
            results = await asyncio.gather(*tasks)

        # Return first non-None result
        for value in results:
            if value is not None:
                return value

        return None

    async def _execute_parallel(
        self,
        tasks: list,
    ) -> list[tuple[bool, Any]]:
        """Execute tasks in parallel and return results.

        For ALL_MUST_PASS: cancels remaining tasks on first failure.
        For FIRST_SUCCESS/RACE: cancels remaining tasks on first success.
        For COLLECT_ALL: waits for all tasks to complete.

        Args:
            tasks: List of coroutines to execute.

        Returns:
            List of (success, result_or_exception) tuples.

        Raises:
            asyncio.TimeoutError: If timeout is exceeded.
        """
        if not tasks:
            return []  # pragma: no cover

        # For COLLECT_ALL, we need all results - use gather
        if self.strategy == AggregationStrategy.COLLECT_ALL:
            return await self._execute_all(tasks)

        # For other strategies, use early cancellation
        return await self._execute_with_early_cancel(tasks)

    async def _execute_all(self, tasks: list) -> list[tuple[bool, Any]]:
        """Execute all tasks and wait for completion."""
        if self.timeout is not None:
            raw_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout,
            )
        else:
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        return [(not isinstance(r, BaseException), r) for r in raw_results]

    async def _execute_with_early_cancel(
        self,
        tasks: list,
    ) -> list[tuple[bool, Any]]:
        """Execute tasks with early cancellation based on strategy.

        ALL_MUST_PASS: Cancel remaining on first failure (but wait for all if no failures).
        FIRST_SUCCESS: Cancel remaining on first success.
        RACE: Cancel remaining after first completion (success or failure).
        """
        # Wrap coroutines in tasks so we can cancel them
        pending: set[asyncio.Task[Any]] = {
            asyncio.create_task(coro) for coro in tasks
        }
        results: list[tuple[bool, Any]] = []
        should_cancel = False

        try:
            deadline = (
                asyncio.get_event_loop().time() + self.timeout
                if self.timeout
                else None
            )

            while pending:
                # Calculate remaining timeout
                timeout = None
                if deadline is not None:
                    timeout = max(0, deadline - asyncio.get_event_loop().time())
                    if timeout == 0:  # pragma: no cover
                        raise asyncio.TimeoutError()

                done, pending = await asyncio.wait(
                    pending,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done and timeout is not None:  # pragma: no cover
                    raise asyncio.TimeoutError()

                for task in done:
                    try:
                        result = task.result()
                        results.append((True, result))

                        # FIRST_SUCCESS or RACE: cancel on first success
                        if self.strategy in (
                            AggregationStrategy.FIRST_SUCCESS,
                            AggregationStrategy.RACE,
                        ):
                            should_cancel = True

                    except Exception as e:
                        results.append((False, e))

                        # ALL_MUST_PASS: cancel on first failure
                        if self.strategy == AggregationStrategy.ALL_MUST_PASS:
                            should_cancel = True

                        # RACE: cancel on first completion (even failure)
                        if self.strategy == AggregationStrategy.RACE:
                            should_cancel = True

                # Cancel after processing all completed tasks in this batch
                if should_cancel and pending:
                    self._cancel_tasks(pending)
                    break

            return results

        except asyncio.TimeoutError:
            self._cancel_tasks(pending)
            raise

    def _cancel_tasks(self, tasks: set[asyncio.Task[Any]]) -> None:
        """Cancel all pending tasks."""
        for task in tasks:
            task.cancel()

    def _aggregate_results(
        self,
        results: list[tuple[bool, Any]],
        default_value: Any,
    ) -> Any:
        """Aggregate results based on the configured strategy.

        Args:
            results: List of (success, result_or_exception) tuples.
            default_value: Value to return if no results match criteria.

        Returns:
            Aggregated result based on strategy.

        Raises:
            ParallelExecutionFailed: If ALL_MUST_PASS and any middleware failed.
            AggregationFailed: If aggregation cannot produce a valid result.
        """
        successes: list[Any] = []
        failures: list[Exception] = []

        for success, value in results:
            if success:
                successes.append(value)
            else:
                failures.append(value)

        if self.strategy == AggregationStrategy.ALL_MUST_PASS:
            # All must succeed - if any failed, raise the first failure
            if failures:
                # Raise the first error directly for cleaner stack traces
                raise failures[0]
            # All succeeded - return the last successful result
            # (preserves behavior similar to sequential middleware)
            return successes[-1] if successes else default_value

        elif self.strategy == AggregationStrategy.FIRST_SUCCESS:
            # Return first success, or raise aggregation error if all failed
            if successes:
                return successes[0]
            if failures:
                raise AggregationFailed(
                    strategy=self.strategy,
                    reason="All middleware failed",
                    errors=failures,
                )
            return default_value  # pragma: no cover

        elif self.strategy == AggregationStrategy.RACE:
            # Return first result (in order), whether success or failure
            if results:
                success, value = results[0]
                if not success:
                    raise value
                return value
            return default_value  # pragma: no cover

        elif self.strategy == AggregationStrategy.COLLECT_ALL:
            # Return all results as-is (caller handles the list)
            return results

        # Should never reach here, but satisfy type checker
        return default_value  # pragma: no cover

"""Interactive demo: Parallel vs Sequential validation.

Run with: uv run python examples/parallel_validation.py
"""

import asyncio
import re
import time
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    AggregationStrategy,
    InputBlocked,
    MiddlewareAgent,
    ParallelMiddleware,
)


class ProfanityFilter(AgentMiddleware[None]):
    """Checks for blocked words (simulates 0.3s API call)."""

    async def before_run(self, prompt: str | Sequence[Any], deps: None) -> str | Sequence[Any]:
        await asyncio.sleep(0.3)
        if isinstance(prompt, str):
            blocked = {"badword", "offensive"}
            if set(prompt.lower().split()) & blocked:
                raise InputBlocked("Profanity detected")
        return prompt


class PIIDetector(AgentMiddleware[None]):
    """Checks for email/phone (simulates 1.0s API call)."""

    async def before_run(self, prompt: str | Sequence[Any], deps: None) -> str | Sequence[Any]:
        await asyncio.sleep(5.0)
        if isinstance(prompt, str):
            if re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", prompt):
                raise InputBlocked("Email address detected")
            if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", prompt):
                raise InputBlocked("Phone number detected")
        return prompt


class InjectionGuard(AgentMiddleware[None]):
    """Checks for prompt injection (simulates 0.5s API call)."""

    async def before_run(self, prompt: str | Sequence[Any], deps: None) -> str | Sequence[Any]:
        await asyncio.sleep(0.5)
        if isinstance(prompt, str):
            patterns = ["ignore previous", "disregard", "you are now"]
            if any(p in prompt.lower() for p in patterns):
                raise InputBlocked("Prompt injection detected")
        return prompt


def create_model() -> TestModel:
    model = TestModel()
    model.custom_output_text = "Hello! How can I help you today?"
    return model


async def run_sequential(prompt: str) -> tuple[str | None, float, str | None]:
    """Run with sequential middleware."""
    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[ProfanityFilter(), PIIDetector(), InjectionGuard()],
    )
    start = time.time()
    try:
        result = await agent.run(prompt)
        return result.output, time.time() - start, None
    except InputBlocked as e:
        return None, time.time() - start, e.reason


async def run_parallel(
    prompt: str, strategy: AggregationStrategy
) -> tuple[str | None, float, str | None]:
    """Run with parallel middleware."""
    parallel = ParallelMiddleware(
        middleware=[ProfanityFilter(), PIIDetector(), InjectionGuard()],
        strategy=strategy,
    )
    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[parallel],
    )
    start = time.time()
    try:
        result = await agent.run(prompt)
        return result.output, time.time() - start, None
    except InputBlocked as e:
        return None, time.time() - start, e.reason


async def main() -> None:
    print("=" * 50)
    print("PARALLEL VALIDATION DEMO")
    print("=" * 50)
    print("\n3 validators run on each input:")
    print("  - ProfanityFilter  0.3s (blocks: 'badword', 'offensive')")
    print("  - PIIDetector      5.0s (blocks: emails, phone numbers)")
    print("  - InjectionGuard   0.5s (blocks: 'ignore previous', etc.)")
    print("\nStrategies:")
    print("  ALL_MUST_PASS  - All must succeed, cancel on first failure")
    print("  FIRST_SUCCESS  - Return on first success, cancel remaining")
    print("\nCommands:")
    print("  /parallel on   - Enable parallel mode")
    print("  /parallel off  - Disable parallel mode (sequential)")
    print("  /strategy all  - Use ALL_MUST_PASS strategy")
    print("  /strategy first - Use FIRST_SUCCESS strategy")
    print("  /strategy race  - Use RACE strategy")
    print("  /quit          - Exit")
    print()

    parallel_mode = True
    strategy = AggregationStrategy.ALL_MUST_PASS

    while True:
        try:
            strategy_name = strategy.value.upper()
            mode = "PARALLEL" if parallel_mode else "SEQUENTIAL"
            prompt = input(f"[{mode} | {strategy_name}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "/quit":
            print("Bye!")
            break

        if prompt.lower() == "/parallel on":
            parallel_mode = True
            print("✓ Parallel mode enabled\n")
            continue

        if prompt.lower() == "/parallel off":
            parallel_mode = False
            print("✓ Sequential mode enabled\n")
            continue

        if prompt.lower() == "/strategy all":
            strategy = AggregationStrategy.ALL_MUST_PASS
            print("✓ Using ALL_MUST_PASS (cancel on first failure)\n")
            continue

        if prompt.lower() == "/strategy first":
            strategy = AggregationStrategy.FIRST_SUCCESS
            print("✓ Using FIRST_SUCCESS (cancel on first success)\n")
            continue

        if prompt.lower() == "/strategy race":
            strategy = AggregationStrategy.RACE
            print("✓ Using RACE (first to finish wins)\n")
            continue

        if parallel_mode:
            output, elapsed, error = await run_parallel(prompt, strategy)
        else:
            output, elapsed, error = await run_sequential(prompt)

        if error:
            print(f"  🚫 BLOCKED: {error}")
        else:
            print(f"  ✓ Response: {output}")
        print(f"  ⏱  Time: {elapsed:.2f}s\n")


if __name__ == "__main__":
    asyncio.run(main())

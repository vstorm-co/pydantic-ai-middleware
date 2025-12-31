"""Interactive demo: Async guardrail timing modes.

Run with: uv run python examples/async_guardrails.py
"""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    AsyncGuardrailMiddleware,
    GuardrailTiming,
    InputBlocked,
    MiddlewareAgent,
)


class SlowSafetyCheck(AgentMiddleware[None]):
    """Simulates a slow ML-based safety check."""

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay

    async def before_run(self, prompt: str | Sequence[Any], deps: None) -> str | Sequence[Any]:
        await asyncio.sleep(self.delay)
        if isinstance(prompt, str) and "dangerous" in prompt.lower():
            raise InputBlocked("Dangerous content detected")
        return prompt


def create_model() -> TestModel:
    model = TestModel()
    model.custom_output_text = "Here's my response to your query!"
    return model


async def run_blocking(prompt: str, guardrail_delay: float) -> tuple[str | None, float, str | None]:
    """BLOCKING: Guardrail finishes before LLM starts."""
    guardrail = AsyncGuardrailMiddleware(
        guardrail=SlowSafetyCheck(delay=guardrail_delay),
        timing=GuardrailTiming.BLOCKING,
    )
    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[guardrail],
    )
    start = time.time()
    try:
        result = await agent.run(prompt)
        return result.output, time.time() - start, None
    except InputBlocked as e:
        return None, time.time() - start, e.reason


async def run_concurrent(
    prompt: str, guardrail_delay: float
) -> tuple[str | None, float, str | None]:
    """CONCURRENT: Guardrail runs alongside LLM, can cancel on failure."""
    guardrail = AsyncGuardrailMiddleware(
        guardrail=SlowSafetyCheck(delay=guardrail_delay),
        timing=GuardrailTiming.CONCURRENT,
        cancel_on_failure=True,
    )
    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[guardrail],
    )
    start = time.time()
    try:
        result = await agent.run(prompt)
        return result.output, time.time() - start, None
    except InputBlocked as e:
        return None, time.time() - start, e.reason


async def main() -> None:
    print("=" * 50)
    print("ASYNC GUARDRAILS DEMO")
    print("=" * 50)
    print("\nA safety guardrail checks each input.")
    print("Include 'dangerous' in your message to trigger blocking.")
    print("\nTiming modes:")
    print("  BLOCKING   - Guardrail runs first, then LLM")
    print("  CONCURRENT - Guardrail + LLM run in parallel")
    print("\nCommands:")
    print("  /mode blocking   - Switch to BLOCKING mode")
    print("  /mode concurrent - Switch to CONCURRENT mode")
    print("  /mode async_post - Switch to ASYNC_POST mode")
    print("  /delay <sec>     - Set guardrail delay (e.g., /delay 0.5)")
    print("  /quit            - Exit")
    print()

    mode: GuardrailTiming = GuardrailTiming.BLOCKING
    guardrail_delay = 0.5

    while True:
        try:
            mode_name = (
                "BLOCKING"
                if mode == GuardrailTiming.BLOCKING
                else "CONCURRENT"
                if mode == GuardrailTiming.CONCURRENT
                else "ASYNC_POST"
            )
            prompt = input(f"[{mode_name} | {guardrail_delay}s] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "/quit":
            print("Bye!")
            break

        if prompt.lower() == "/mode blocking":
            mode = GuardrailTiming.BLOCKING
            print("✓ Switched to BLOCKING mode\n")
            continue

        if prompt.lower() == "/mode concurrent":
            mode = GuardrailTiming.CONCURRENT
            print("✓ Switched to CONCURRENT mode\n")
            continue

        if prompt.lower().startswith("/delay "):
            try:
                guardrail_delay = float(prompt.split()[1])
                print(f"✓ Guardrail delay set to {guardrail_delay}s\n")
            except (IndexError, ValueError):
                print("Usage: /delay <seconds>\n")
            continue

        if mode == GuardrailTiming.BLOCKING:
            output, elapsed, error = await run_blocking(prompt, guardrail_delay)
        else:
            output, elapsed, error = await run_concurrent(prompt, guardrail_delay)

        if error:
            print(f"  🚫 BLOCKED: {error}")
        else:
            print(f"  ✓ Response: {output}")
        print(f"  ⏱  Time: {elapsed:.2f}s\n")


if __name__ == "__main__":
    asyncio.run(main())

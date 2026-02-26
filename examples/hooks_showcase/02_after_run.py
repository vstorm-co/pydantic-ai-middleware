"""Example: after_run hook — called AFTER the agent finishes.

Runs in REVERSE middleware order. Use it to:
- Log the final output
- Transform or filter responses
- Record analytics / timing

Run: uv run python examples/hooks_showcase/02_after_run.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class TimingMiddleware(AgentMiddleware[None]):
    """Measures wall-clock time and logs the final output."""

    def __init__(self) -> None:
        self._start: float = 0

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        self._start = time.perf_counter()
        print("[before_run] Timer started")
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        elapsed = time.perf_counter() - self._start
        print(f"[after_run] Output: {output!r}")
        print(f"[after_run] Elapsed: {elapsed:.3f}s")
        return output


async def main() -> None:
    agent = Agent("openai:gpt-4.1", instructions="Answer in one short sentence.")
    mw_agent = MiddlewareAgent(agent=agent, middleware=[TimingMiddleware()])

    result = await mw_agent.run("What is 2 + 2?")
    print(f"[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

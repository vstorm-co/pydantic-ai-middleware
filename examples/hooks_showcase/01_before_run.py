"""Example: before_run hook — called BEFORE the agent starts processing.

This is the first hook in the middleware pipeline. Use it to:
- Log incoming prompts
- Modify or sanitize user input
- Block certain inputs (raise InputBlocked)

Run: uv run python examples/hooks_showcase/01_before_run.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class BeforeRunLogger(AgentMiddleware[None]):
    """Logs every prompt before the agent processes it."""

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        print(f"[before_run] Received prompt: {prompt!r}")
        return prompt


async def main() -> None:
    agent = Agent("openai:gpt-4.1", instructions="Answer in one short sentence.")
    mw_agent = MiddlewareAgent(agent=agent, middleware=[BeforeRunLogger()])

    result = await mw_agent.run("What is the capital of France?")
    print(f"[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

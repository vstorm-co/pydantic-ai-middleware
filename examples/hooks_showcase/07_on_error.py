"""Example: on_error hook — called when ANY exception occurs during the run.

Use it to:
- Log all errors globally
- Replace exceptions with custom error types
- Send error reports to monitoring services

Run: uv run python examples/hooks_showcase/07_on_error.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent
from pydantic_ai_middleware.exceptions import InputBlocked


class GlobalErrorLogger(AgentMiddleware[None]):
    """Catches and logs any exception that occurs during the agent run."""

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        # Simulate blocking a dangerous prompt
        if isinstance(prompt, str) and "hack" in prompt.lower():
            raise InputBlocked("Prompt contains blocked content")
        return prompt

    async def on_error(
        self,
        error: Exception,
        deps: None,
        ctx: Any = None,
    ) -> Exception | None:
        print(f"[on_error] Caught: {type(error).__name__}: {error}")
        # Return None to re-raise original, or return a new exception
        return None


async def main() -> None:
    agent = Agent("openai:gpt-4.1", instructions="Answer briefly.")
    mw_agent = MiddlewareAgent(agent=agent, middleware=[GlobalErrorLogger()])

    # This will trigger InputBlocked -> on_error
    try:
        await mw_agent.run("How to hack a system?")
    except InputBlocked as e:
        print(f"[main] Blocked: {e.reason}")

    print()

    # This will work normally
    result = await mw_agent.run("What is Python?")
    print(f"[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

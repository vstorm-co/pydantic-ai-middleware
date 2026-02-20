"""Example: MiddlewareChain — compose multiple middleware into a reusable unit.

Chains maintain execution order: before_* hooks run first→last,
after_* hooks run last→first (LIFO).

Run: uv run python examples/hooks_showcase/11_middleware_chain.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent, MiddlewareChain


class LoggingMiddleware(AgentMiddleware[None]):
    """Logs before and after run."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        print(f"[{self.name}] before_run")
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        print(f"[{self.name}] after_run")
        return output


async def main() -> None:
    agent = Agent("openai:gpt-4.1", instructions="Answer in one word.")

    # Create a reusable chain of middleware
    security_chain = MiddlewareChain(
        [
            LoggingMiddleware("Auth"),
            LoggingMiddleware("RateLimit"),
        ],
        name="SecurityChain",
    )

    observability_chain = MiddlewareChain(
        [
            LoggingMiddleware("Metrics"),
            LoggingMiddleware("Tracing"),
        ],
        name="ObservabilityChain",
    )

    # Compose chains together
    mw_agent = MiddlewareAgent(
        agent=agent,
        middleware=[security_chain, observability_chain],
    )

    print("=== Execution order ===")
    result = await mw_agent.run("Hello!")
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

"""Example: before_model_request hook — called BEFORE each LLM API call.

Injected as a history processor. Called multiple times per run
(once per model round-trip). Use it to:
- Log or inspect the message history
- Inject system messages
- Count model calls

Run: uv run python examples/hooks_showcase/03_before_model_request.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class ModelRequestLogger(AgentMiddleware[None]):
    """Logs message count before each model API call."""

    def __init__(self) -> None:
        self.call_count = 0

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: None,
        ctx: Any = None,
    ) -> list[ModelMessage]:
        self.call_count += 1
        print(f"[before_model_request] Call #{self.call_count}")
        print(f"[before_model_request] Messages in history: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"  [{i}] {type(msg).__name__}")
        return messages


async def main() -> None:
    agent = Agent("openai:gpt-4.1", instructions="Answer in one short sentence.")
    mw = ModelRequestLogger()
    mw_agent = MiddlewareAgent(agent=agent, middleware=[mw])

    result = await mw_agent.run("Explain what middleware is")
    print(f"\n[result] {result.output}")
    print(f"[stats] Total model calls: {mw.call_count}")


if __name__ == "__main__":
    asyncio.run(main())

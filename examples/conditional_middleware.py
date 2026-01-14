"""Demo: conditional middleware driven by MiddlewareContext.

Run with: uv run python examples/conditional_middleware.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    ConditionalMiddleware,
    InputBlocked,
    MiddlewareAgent,
    MiddlewareContext,
    ScopedContext,
)


class MinLengthValidator(AgentMiddleware[None]):
    """Reject prompts that are too short."""

    def __init__(self, min_length: int = 5) -> None:
        self.min_length = min_length

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        if isinstance(prompt, str) and len(prompt) < self.min_length:
            raise InputBlocked(f"Prompt too short (< {self.min_length})")
        return prompt


def guardrails_enabled(ctx: ScopedContext | None) -> bool:
    return bool(ctx and ctx.config.get("guardrails"))


def create_model() -> TestModel:
    model = TestModel()
    model.custom_output_text = "ok"
    return model


async def run_with_context(config: dict[str, Any], prompt: str) -> None:
    guardrails = ConditionalMiddleware(
        condition=guardrails_enabled,
        when_true=MinLengthValidator(min_length=5),
    )

    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[guardrails],
        context=MiddlewareContext(config=config),
    )

    try:
        result = await agent.run(prompt)
        print(f"Result: {result.output}")
    except InputBlocked as exc:
        print(f"Blocked: {exc}")


async def main() -> None:
    print("Guardrails enabled:")
    await run_with_context({"guardrails": True}, "hi")

    print("\nGuardrails disabled:")
    await run_with_context({"guardrails": False}, "hi")


if __name__ == "__main__":
    asyncio.run(main())

"""Basic usage — combine multiple shields on one agent."""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_shields import (
    CostTracking,
    NoRefusals,
    PiiDetector,
    PromptInjection,
    SecretRedaction,
)


async def main() -> None:
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            CostTracking(budget_usd=5.0),
            PromptInjection(sensitivity="high"),
            PiiDetector(),
            SecretRedaction(),
            NoRefusals(),
        ],
    )

    result = await agent.run("What is the capital of France?")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

"""Example: load and save middleware pipelines from config."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareAgent,
    load_middleware_config_text,
    save_middleware_config_path,
)


class LoggingMiddleware(AgentMiddleware[None]):
    """Log inputs and outputs."""

    async def before_run(self, prompt, deps, ctx):
        print(f"[log] input: {prompt}")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        print(f"[log] output: {output}")
        return output


class ProfanityFilter(AgentMiddleware[None]):
    """Redact a simple set of words."""

    def __init__(self) -> None:
        self._blocked = {"bad", "naughty"}

    async def before_run(self, prompt, deps, ctx):
        text = str(prompt)
        for word in self._blocked:
            text = text.replace(word, "***")
        return text


async def main() -> None:
    registry = {
        "logging": LoggingMiddleware,
        "profanity": ProfanityFilter,
    }

    config_text = """
    [
      {"type": "logging"},
      {"type": "profanity"}
    ]
    """

    middleware = load_middleware_config_text(config_text, registry=registry)
    save_middleware_config_path(
        [
            {"type": "logging"},
            {"type": "profanity"},
        ],
        "examples/pipeline.json",
    )

    base_agent = Agent(model=TestModel())
    agent = MiddlewareAgent(agent=base_agent, middleware=middleware)
    result = await agent.run("Hello bad world")
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

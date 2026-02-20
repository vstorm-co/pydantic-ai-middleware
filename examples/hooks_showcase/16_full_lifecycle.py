"""Example: Full lifecycle — ALL hooks in one middleware.

Demonstrates the complete execution order:
  before_run → before_model_request → before_tool_call →
  [tool executes] → after_tool_call → before_model_request (2nd call) →
  after_run

Run: uv run python examples/hooks_showcase/16_full_lifecycle.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class FullLifecycleLogger(AgentMiddleware[None]):
    """Implements ALL hooks to show the complete execution flow."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def _log(self, event: str) -> None:
        self.events.append(event)
        step = len(self.events)
        print(f"  {step}. {event}")

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        self._log(f"before_run(prompt={prompt!r})")
        return prompt

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: None,
        ctx: Any = None,
    ) -> list[ModelMessage]:
        self._log(f"before_model_request(messages={len(messages)})")
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        self._log(f"before_tool_call(tool={tool_name}, args={tool_args})")
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        self._log(f"after_tool_call(tool={tool_name}, result={result!r})")
        return result

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        self._log(f"after_run(output={str(output)[:60]!r})")
        return output

    async def on_error(
        self,
        error: Exception,
        deps: None,
        ctx: Any = None,
    ) -> Exception | None:
        self._log(f"on_error({type(error).__name__}: {error})")
        return None


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the calculate tool to answer math questions. Be brief.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    result = eval(expression)  # noqa: S307
    return str(result)


async def main() -> None:
    mw = FullLifecycleLogger()
    mw_agent = MiddlewareAgent(agent=agent, middleware=[mw])

    print("=== Full Lifecycle Trace ===\n")
    result = await mw_agent.run("What is 7 * 8 + 2?", toolsets=[toolset])
    print(f"\n[result] {result.output}")
    print(f"\n=== Total events: {len(mw.events)} ===")


if __name__ == "__main__":
    asyncio.run(main())

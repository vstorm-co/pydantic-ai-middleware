"""Example: Multiple middleware executing in order.

Shows the pipeline execution order:
- before_* hooks: run in ORDER (first → last)
- after_* hooks: run in REVERSE order (last → first)

This is the same pattern as ASGI/WSGI middleware stacking.

Run: uv run python examples/hooks_showcase/15_multiple_middleware.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class NumberedMiddleware(AgentMiddleware[None]):
    """Middleware that prints its position to show execution order."""

    def __init__(self, number: int) -> None:
        self.number = number

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        print(f"  [MW-{self.number}] before_run")
        return prompt

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        print(f"  [MW-{self.number}] before_tool_call({tool_name})")
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        print(f"  [MW-{self.number}] after_tool_call({tool_name})")
        return result

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        print(f"  [MW-{self.number}] after_run")
        return output


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the echo tool to answer. Be brief.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def echo(text: str) -> str:
    """Echo the given text."""
    print(f"  [tool:echo] '{text}'")
    return text


async def main() -> None:
    mw_agent = MiddlewareAgent(
        agent=agent,
        middleware=[
            NumberedMiddleware(1),
            NumberedMiddleware(2),
            NumberedMiddleware(3),
        ],
    )

    print("=== Expected: before_* = 1,2,3  |  after_* = 3,2,1 ===\n")
    result = await mw_agent.run("Echo 'hello world'", toolsets=[toolset])
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

"""Example: before_tool_call hook — called BEFORE each tool execution.

Use it to:
- Log tool invocations and arguments
- Modify tool arguments (e.g. sanitize)
- Block tools (raise ToolBlocked or return ToolPermissionResult)

Run: uv run python examples/hooks_showcase/04_before_tool_call.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class BeforeToolCallLogger(AgentMiddleware[None]):
    """Logs every tool call before it executes."""

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        print(f"[before_tool_call] Tool: {tool_name}")
        print(f"[before_tool_call] Args: {tool_args}")
        return tool_args


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the add tool to answer math questions. Answer briefly.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def add(a: int, b: int) -> str:
    """Add two numbers together."""
    result = a + b
    print(f"  [tool:add] Computing {a} + {b} = {result}")
    return str(result)


async def main() -> None:
    mw_agent = MiddlewareAgent(agent=agent, middleware=[BeforeToolCallLogger()])

    result = await mw_agent.run("What is 17 + 25?", toolsets=[toolset])
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

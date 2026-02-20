"""Example: on_tool_error hook — called when a tool raises an exception.

Use it to:
- Log tool failures with full context (tool name + args)
- Replace errors with friendlier messages
- Return None to re-raise the original exception

Run: uv run python examples/hooks_showcase/06_on_tool_error.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class ToolErrorHandler(AgentMiddleware[None]):
    """Logs tool errors and replaces them with user-friendly messages."""

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: None,
        ctx: Any = None,
    ) -> Exception | None:
        print(f"[on_tool_error] Tool '{tool_name}' failed!")
        print(f"[on_tool_error] Args: {tool_args}")
        print(f"[on_tool_error] Error: {type(error).__name__}: {error}")
        # Replace with a cleaner error
        return RuntimeError(f"Tool '{tool_name}' is temporarily unavailable")


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the divide tool to answer. If it fails, say so.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def divide(a: int, b: int) -> str:
    """Divide two numbers."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return str(a / b)


async def main() -> None:
    mw_agent = MiddlewareAgent(agent=agent, middleware=[ToolErrorHandler()])

    result = await mw_agent.run("What is 10 divided by 0?", toolsets=[toolset])
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

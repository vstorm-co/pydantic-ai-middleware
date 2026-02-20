"""Example: after_tool_call hook — called AFTER each tool execution.

Runs in REVERSE middleware order. Use it to:
- Log tool results
- Modify or enrich tool output
- Record performance metrics per tool

Run: uv run python examples/hooks_showcase/05_after_tool_call.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class AfterToolCallLogger(AgentMiddleware[None]):
    """Logs tool results and measures execution time."""

    def __init__(self) -> None:
        self._tool_starts: dict[str, float] = {}

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        self._tool_starts[tool_name] = time.perf_counter()
        return tool_args

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        elapsed = time.perf_counter() - self._tool_starts.pop(tool_name, 0)
        print(f"[after_tool_call] Tool: {tool_name}")
        print(f"[after_tool_call] Result: {result!r}")
        print(f"[after_tool_call] Took: {elapsed:.4f}s")
        return result


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the multiply tool to answer. Answer briefly.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    result = a * b
    print(f"  [tool:multiply] {a} * {b} = {result}")
    return str(result)


async def main() -> None:
    mw_agent = MiddlewareAgent(agent=agent, middleware=[AfterToolCallLogger()])

    result = await mw_agent.run("What is 12 times 8?", toolsets=[toolset])
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

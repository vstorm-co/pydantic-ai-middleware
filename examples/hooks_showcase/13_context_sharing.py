"""Example: MiddlewareContext — share data between hooks.

Context provides scoped, access-controlled storage:
- Each hook can WRITE to its own namespace
- Each hook can READ from earlier hooks' namespaces
- Access control prevents reading from hooks that run later

Run: uv run python examples/hooks_showcase/13_context_sharing.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent, MiddlewareContext
from pydantic_ai_middleware.context import HookType


class RequestTracker(AgentMiddleware[None]):
    """Stores data in context and reads it across hooks."""

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: Any = None,
    ) -> str | Sequence[Any]:
        if ctx:
            ctx.set("start_time", time.perf_counter())
            ctx.set("prompt_length", len(str(prompt)))
            print(f"[before_run] Stored start_time and prompt_length in context")
        return prompt

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        if ctx:
            # Read data from before_run hook
            prompt_len = ctx.get_from(HookType.BEFORE_RUN, "prompt_length")
            print(f"[before_tool_call] Read from before_run: prompt_length={prompt_len}")
            ctx.set("tool_name", tool_name)
        return tool_args

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: Any = None,
    ) -> Any:
        if ctx:
            start = ctx.get_from(HookType.BEFORE_RUN, "start_time", 0)
            elapsed = time.perf_counter() - start
            print(f"[after_run] Total elapsed: {elapsed:.3f}s")

            # Read from before_tool_call
            tool = ctx.get_from(HookType.BEFORE_TOOL_CALL, "tool_name", "none")
            print(f"[after_run] Last tool used: {tool}")
        return output


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the lookup tool to answer. Be brief.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def lookup(query: str) -> str:
    """Look up information."""
    return f"Result for '{query}': 42"


async def main() -> None:
    # MiddlewareContext enables the ctx parameter
    context = MiddlewareContext(config={"environment": "demo"})

    mw_agent = MiddlewareAgent(
        agent=agent,
        middleware=[RequestTracker()],
        context=context,
    )

    result = await mw_agent.run("Look up the meaning of life", toolsets=[toolset])
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

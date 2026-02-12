"""Example: Hook timeouts - prevent slow middleware from blocking the agent.

This example demonstrates how to set per-middleware timeouts and
handle MiddlewareTimeout exceptions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai_middleware import AgentMiddleware, MiddlewareTimeout, ScopedContext


class ExternalAPICheck(AgentMiddleware[None]):
    """Check input against an external moderation API with a timeout."""

    timeout = 3.0  # 3 seconds max for all hooks

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        # Simulates a slow external API call
        await asyncio.sleep(0.01)  # In reality this could be slow
        return prompt


class SlowToolGuard(AgentMiddleware[None]):
    """Guard specific tools with a per-middleware timeout."""

    tool_names = {"web_search"}
    timeout = 5.0

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        # Validate search query against a policy service
        await asyncio.sleep(0.01)
        return tool_args

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        # Log error to monitoring service
        await asyncio.sleep(0.01)
        return None


# --- Handling MiddlewareTimeout ---


async def demo_timeout_handling() -> None:
    """Show how to catch and handle MiddlewareTimeout."""
    try:
        # This would normally be: result = await agent.run("test")
        # Simulating what happens when a middleware times out:
        raise MiddlewareTimeout("ExternalAPICheck", 3.0, "before_run")
    except MiddlewareTimeout as e:
        print(f"Middleware '{e.middleware_name}' timed out")
        print(f"Hook: {e.hook_name}")
        print(f"Timeout: {e.timeout}s")
        print(f"Full message: {e}")


if __name__ == "__main__":
    asyncio.run(demo_timeout_handling())

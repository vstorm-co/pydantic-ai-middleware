"""Example: Blocking tools with ToolPermissionResult.

Shows the three permission decisions:
- ALLOW: let the tool call proceed (optionally with modified args)
- DENY: block the tool call with a reason
- ASK: defer to a permission_handler callback

Run: uv run python examples/hooks_showcase/08_tool_blocking.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareAgent,
    ToolDecision,
    ToolPermissionResult,
)


class PathGuard(AgentMiddleware[None]):
    """Blocks file operations on sensitive paths."""

    BLOCKED_PATTERNS = ["/etc/", ".env", ".ssh/", "/root/"]

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any] | ToolPermissionResult:
        path = tool_args.get("path", "")
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in path:
                print(f"[PathGuard] DENIED: {tool_name}({path!r})")
                return ToolPermissionResult(
                    decision=ToolDecision.DENY,
                    reason=f"Access to '{pattern}' is not allowed",
                )

        print(f"[PathGuard] ALLOWED: {tool_name}({path!r})")
        return ToolPermissionResult(decision=ToolDecision.ALLOW)


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use read_file to answer. If blocked, explain that access was denied.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    print(f"  [tool:read_file] Reading {path}")
    return f"Contents of {path}: [mock data]"


async def main() -> None:
    mw_agent = MiddlewareAgent(agent=agent, middleware=[PathGuard()])

    # Safe path — should work
    print("--- Safe path ---")
    result = await mw_agent.run("Read the file /home/user/notes.txt", toolsets=[toolset])
    print(f"[result] {result.output}\n")

    # Sensitive path — should be blocked
    print("--- Blocked path ---")
    result = await mw_agent.run("Read the file /etc/passwd", toolsets=[toolset])
    print(f"[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

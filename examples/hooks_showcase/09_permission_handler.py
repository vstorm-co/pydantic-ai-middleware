"""Example: ASK decision with permission_handler callback.

When middleware returns ToolDecision.ASK, the MiddlewareAgent calls
the permission_handler to decide. This enables interactive approval flows.

Run: uv run python examples/hooks_showcase/09_permission_handler.py
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


class InteractiveGuard(AgentMiddleware[None]):
    """Asks for permission before executing destructive tools."""

    DESTRUCTIVE_TOOLS = {"delete_file", "write_file"}

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any] | ToolPermissionResult:
        if tool_name in self.DESTRUCTIVE_TOOLS:
            return ToolPermissionResult(
                decision=ToolDecision.ASK,
                reason=f"'{tool_name}' is destructive — requires approval",
            )
        return tool_args


agent = Agent(
    "openai:gpt-4.1",
    instructions=(
        "Use the available tools to answer. If a tool is blocked, explain what happened."
    ),
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def delete_file(path: str) -> str:
    """Delete a file."""
    print(f"  [tool:delete_file] Deleting {path}")
    return f"Deleted {path}"


@toolset.tool
def list_files(directory: str) -> str:
    """List files in a directory."""
    print(f"  [tool:list_files] Listing {directory}")
    return "file1.txt, file2.txt, secret.env"


async def auto_deny_handler(tool_name: str, tool_args: dict[str, Any], reason: str) -> bool:
    """Simulated permission handler — auto-denies destructive actions."""
    print(f"[permission_handler] Tool: {tool_name}, Reason: {reason}")
    print("[permission_handler] Decision: DENIED (auto)")
    return False  # Deny


async def main() -> None:
    mw_agent = MiddlewareAgent(
        agent=agent,
        middleware=[InteractiveGuard()],
        permission_handler=auto_deny_handler,
    )

    # list_files is safe — goes through
    print("--- Safe tool ---")
    result = await mw_agent.run("List files in /home/user/", toolsets=[toolset])
    print(f"[result] {result.output}\n")

    # delete_file is destructive — triggers ASK → handler denies
    print("--- Destructive tool ---")
    result = await mw_agent.run("Delete the file /home/user/secret.env", toolsets=[toolset])
    print(f"[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

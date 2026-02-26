"""Example: tool_names filtering — middleware that only runs for specific tools.

Set `tool_names = {"tool_a", "tool_b"}` on your middleware class to limit
which tools trigger before_tool_call / after_tool_call / on_tool_error.

Run: uv run python examples/hooks_showcase/10_tool_name_filtering.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent


class EmailOnlyGuard(AgentMiddleware[None]):
    """Only fires for email-related tools."""

    tool_names = {"send_email"}

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        print(f"[EmailOnlyGuard] Checking {tool_name}: {tool_args}")
        if not tool_args.get("to"):
            from pydantic_ai_middleware.exceptions import ToolBlocked

            raise ToolBlocked(tool_name, "Recipient (to) is required!")
        return tool_args


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the tools to fulfill requests. Answer briefly.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    print(f"  [tool:send_email] To: {to}, Subject: {subject}")
    return f"Email sent to {to}"


@toolset.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    print(f"  [tool:get_weather] City: {city}")
    return f"Weather in {city}: 22C, sunny"


async def main() -> None:
    mw_agent = MiddlewareAgent(agent=agent, middleware=[EmailOnlyGuard()])

    # get_weather — EmailOnlyGuard does NOT fire (not in tool_names)
    print("--- Weather (no guard) ---")
    result = await mw_agent.run("What's the weather in Warsaw?", toolsets=[toolset])
    print(f"[result] {result.output}\n")

    # send_email — EmailOnlyGuard FIRES
    print("--- Email (guard active) ---")
    result = await mw_agent.run(
        "Send an email to jan@example.com with subject 'Hello' and body 'Hi!'",
        toolsets=[toolset],
    )
    print(f"[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

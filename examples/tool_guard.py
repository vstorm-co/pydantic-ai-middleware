"""Tool guard — block tools or require human approval."""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_shields import ToolBlocked, ToolGuard


async def ask_user(tool_name: str, args: dict) -> bool:
    """Simple approval callback."""
    response = input(f"Allow {tool_name} with {args}? (y/n) ")
    return response.lower() == "y"


async def main() -> None:
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            ToolGuard(
                blocked=["execute", "rm"],
                require_approval=["write_file", "edit_file"],
                approval_callback=ask_user,
            )
        ],
    )

    # Blocked tools are hidden from the model — it won't even try to call them.
    # Approval tools trigger the callback before execution.
    try:
        result = await agent.run("Write hello world to test.py")
        print(result.output)
    except ToolBlocked as e:
        print(f"Tool blocked: {e}")


if __name__ == "__main__":
    asyncio.run(main())

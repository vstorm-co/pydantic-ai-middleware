"""Async guardrail — run safety check concurrently with LLM call."""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_shields import AsyncGuardrail, InputBlocked, InputGuard


async def slow_policy_check(prompt: str) -> bool:
    """Simulate a slow external policy API."""
    await asyncio.sleep(0.5)
    return "forbidden_topic" not in prompt.lower()


async def main() -> None:
    # Concurrent: policy check runs alongside LLM.
    # If policy fails before LLM finishes, result is discarded (saves cost).
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            AsyncGuardrail(
                guard=InputGuard(guard=slow_policy_check),
                timing="concurrent",
                cancel_on_failure=True,
                timeout=5.0,
            )
        ],
    )

    try:
        result = await agent.run("Tell me about forbidden_topic")
    except InputBlocked as e:
        print(f"Blocked by concurrent guard: {e}")

    # Safe prompt passes through
    result = await agent.run("What is the weather?")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

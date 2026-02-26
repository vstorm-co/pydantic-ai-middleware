"""Example: CostTrackingMiddleware — track tokens and USD costs.

Built-in middleware that:
- Counts input/output tokens per run and cumulatively
- Calculates USD costs using genai-prices
- Supports budget limits (raises BudgetExceededError)
- Fires on_cost_update callback after each run

Run: uv run python examples/hooks_showcase/14_cost_tracking.py
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from pydantic_ai_middleware import MiddlewareAgent, MiddlewareContext
from pydantic_ai_middleware.cost_tracking import CostInfo, CostTrackingMiddleware


def on_cost(info: CostInfo) -> None:
    """Callback fired after each run with cost data."""
    print(f"[cost] Run #{info.run_count}")
    print(f"[cost]   Tokens: {info.run_request_tokens} in / {info.run_response_tokens} out")
    if info.run_cost_usd is not None:
        print(f"[cost]   Run cost: ${info.run_cost_usd:.6f}")
        print(f"[cost]   Total cost: ${info.total_cost_usd:.6f}")


async def main() -> None:
    agent = Agent("openai:gpt-4.1", instructions="Answer in one sentence.")

    cost_mw = CostTrackingMiddleware(
        model_name="openai:gpt-4.1",
        budget_limit_usd=1.00,  # $1 budget
        on_cost_update=on_cost,
    )

    mw_agent = MiddlewareAgent(
        agent=agent,
        middleware=[cost_mw],
        context=MiddlewareContext(),  # Required for cost tracking
    )

    # Run 1
    print("=== Run 1 ===")
    result = await mw_agent.run("What is Python?")
    print(f"[result] {result.output}\n")

    # Run 2
    print("=== Run 2 ===")
    result = await mw_agent.run("What is Rust?")
    print(f"[result] {result.output}\n")

    # Check cumulative stats
    print("=== Cumulative Stats ===")
    print(f"Total runs: {cost_mw.run_count}")
    print(f"Total tokens: {cost_mw.total_request_tokens} in / {cost_mw.total_response_tokens} out")
    print(f"Total cost: ${cost_mw.total_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())

"""Cost tracking — token usage monitoring and budget enforcement."""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_shields import BudgetExceededError, CostInfo, CostTracking


async def main() -> None:
    def on_cost(info: CostInfo) -> None:
        print(
            f"Run #{info.run_count}: {info.run_request_tokens} in / {info.run_response_tokens} out"
        )
        if info.run_cost_usd is not None:
            print(f"  Cost: ${info.run_cost_usd:.4f} (total: ${info.total_cost_usd:.4f})")

    tracking = CostTracking(budget_usd=1.0, on_cost_update=on_cost)
    agent = Agent("openai:gpt-4.1", capabilities=[tracking])

    try:
        for i in range(100):
            await agent.run(f"Query {i}: tell me a fun fact")
    except BudgetExceededError as e:
        print(f"\nBudget exceeded: ${e.total_cost:.4f} > ${e.budget:.4f}")
        print(f"Completed {tracking.run_count} runs")


if __name__ == "__main__":
    asyncio.run(main())

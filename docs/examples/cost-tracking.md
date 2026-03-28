# Cost Tracking Example

Track token usage and USD costs across agent runs.

## Basic Tracking

```python
from pydantic_ai import Agent
from pydantic_ai_shields import CostTracking

tracking = CostTracking()
agent = Agent("openai:gpt-4.1", capabilities=[tracking])

await agent.run("Hello")
await agent.run("What is 2+2?")

print(f"Total cost: ${tracking.total_cost:.4f}")
print(f"Total runs: {tracking.run_count}")
```

## With Budget Limit

```python
from pydantic_ai_shields import CostTracking, BudgetExceededError

tracking = CostTracking(budget_usd=1.0)
agent = Agent("openai:gpt-4.1", capabilities=[tracking])

try:
    for i in range(100):
        await agent.run(f"Query {i}")
except BudgetExceededError:
    print(f"Stopped at ${tracking.total_cost:.4f}")
```

## With Callback

```python
from pydantic_ai_shields import CostTracking, CostInfo

def log_cost(info: CostInfo):
    print(f"Run #{info.run_count}: {info.run_request_tokens} in / {info.run_response_tokens} out")
    if info.run_cost_usd is not None:
        print(f"  Cost: ${info.run_cost_usd:.4f} (total: ${info.total_cost_usd:.4f})")

agent = Agent("openai:gpt-4.1", capabilities=[CostTracking(on_cost_update=log_cost)])
```

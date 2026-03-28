# Cost Tracking

`CostTracking` tracks token usage and calculates USD costs across agent runs using
[genai-prices](https://pypi.org/project/genai-prices/).

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_shields import CostTracking

tracking = CostTracking(budget_usd=5.0)
agent = Agent("openai:gpt-4.1", capabilities=[tracking])

result = await agent.run("Hello")

print(f"Cost: ${tracking.total_cost:.4f}")
print(f"Tokens: {tracking.total_request_tokens} in / {tracking.total_response_tokens} out")
print(f"Runs: {tracking.run_count}")
```

## Budget Enforcement

When cumulative cost exceeds `budget_usd`, a `BudgetExceededError` is raised:

```python
from pydantic_ai_shields import CostTracking, BudgetExceededError

tracking = CostTracking(budget_usd=1.0)
agent = Agent("openai:gpt-4.1", capabilities=[tracking])

try:
    for i in range(1000):
        await agent.run(f"Query {i}")
except BudgetExceededError as e:
    print(f"Budget exceeded: ${e.total_cost:.4f} > ${e.budget:.4f}")
```

## Cost Callbacks

```python
from pydantic_ai_shields import CostTracking, CostInfo

def on_cost(info: CostInfo):
    print(f"Run #{info.run_count}: ${info.run_cost_usd:.4f} (total: ${info.total_cost_usd:.4f})")

agent = Agent("openai:gpt-4.1", capabilities=[CostTracking(on_cost_update=on_cost)])
```

## Auto-Detection

Model pricing is auto-detected from `ctx.model.model_id` on the first run.
You can also specify explicitly:

```python
CostTracking(model_name="openai:gpt-4.1", budget_usd=5.0)
```

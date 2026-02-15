# Cost Tracking

`CostTrackingMiddleware` automatically tracks token usage and calculates USD costs
across agent runs. It uses [genai-prices](https://pypi.org/project/genai-prices/) to
look up per-token pricing for supported models.

## Why use cost tracking

- **Visibility** into how many tokens each run consumes.
- **Budget enforcement** to prevent runaway API spend.
- **Real-time callbacks** for dashboards, logging, or alerting.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_middleware import MiddlewareAgent, MiddlewareContext
from pydantic_ai_middleware.cost_tracking import create_cost_tracking_middleware

cost_mw = create_cost_tracking_middleware(
    model_name="openai:gpt-4.1",
    budget_limit_usd=5.0,
    on_cost_update=lambda info: print(
        f"Run #{info.run_count}: ${info.run_cost_usd:.4f} "
        f"(total: ${info.total_cost_usd:.4f})"
    ),
)

agent = MiddlewareAgent(
    agent=Agent(model=TestModel()),
    middleware=[cost_mw],
    context=MiddlewareContext(),
)

result = await agent.run("Summarize this document")
```

!!! note
    A `MiddlewareContext` is required on the `MiddlewareAgent` so the middleware
    can read the `run_usage` metadata that is stored after each run.

## CostInfo fields

After every run the middleware builds a `CostInfo` dataclass and passes it to the
callback. The fields are:

| Field | Type | Description |
|---|---|---|
| `run_cost_usd` | `float | None` | USD cost of this run. `None` if model is unknown. |
| `total_cost_usd` | `float | None` | Cumulative USD cost across all runs. |
| `run_request_tokens` | `int` | Input tokens consumed by this run. |
| `run_response_tokens` | `int` | Output tokens consumed by this run. |
| `total_request_tokens` | `int` | Cumulative input tokens across all runs. |
| `total_response_tokens` | `int` | Cumulative output tokens across all runs. |
| `run_count` | `int` | Number of completed runs so far. |

## Factory function

`create_cost_tracking_middleware()` is a convenience factory that creates a
`CostTrackingMiddleware` instance:

```python
from pydantic_ai_middleware.cost_tracking import create_cost_tracking_middleware

mw = create_cost_tracking_middleware(
    model_name="anthropic:claude-sonnet-4-5-20250929",
    budget_limit_usd=10.0,
    on_cost_update=lambda info: print(f"Total: ${info.total_cost_usd:.4f}"),
)
```

Parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str | None` | `None` | Model identifier in `"provider:model"` format. `None` disables USD cost calculation (tokens are still tracked). |
| `budget_limit_usd` | `float | None` | `None` | Maximum cumulative cost in USD. `None` means unlimited. |
| `on_cost_update` | `CostCallback` | `None` | Sync or async callable invoked after each run with a `CostInfo` object. |

## Budget limits

When `budget_limit_usd` is set the middleware checks cumulative cost in the
`before_run` hook. If the budget has been reached, a `BudgetExceededError` is raised
before the agent processes the prompt.

```python
from pydantic_ai_middleware.cost_tracking import create_cost_tracking_middleware
from pydantic_ai_middleware.exceptions import BudgetExceededError

mw = create_cost_tracking_middleware(
    model_name="openai:gpt-4.1",
    budget_limit_usd=1.0,
)

# ... after many runs ...
try:
    result = await agent.run("Another question")
except BudgetExceededError as e:
    print(f"Budget exceeded: ${e.cost:.4f} >= ${e.budget:.4f}")
```

## Async callbacks

The `on_cost_update` parameter accepts both sync and async callables. The middleware
detects awaitables automatically.

```python
async def save_cost_to_db(info):
    await db.execute(
        "INSERT INTO costs (run, tokens_in, tokens_out, cost) VALUES (?, ?, ?, ?)",
        (info.run_count, info.run_request_tokens, info.run_response_tokens, info.run_cost_usd),
    )

mw = create_cost_tracking_middleware(
    model_name="openai:gpt-4.1",
    on_cost_update=save_cost_to_db,
)
```

## Accessing cumulative state

The middleware instance exposes read-only properties for cumulative totals:

```python
from pydantic_ai_middleware.cost_tracking import CostTrackingMiddleware

mw = CostTrackingMiddleware(model_name="openai:gpt-4.1")

# After running the agent several times...
print(f"Total cost:   ${mw.total_cost:.4f}")
print(f"Total input:  {mw.total_request_tokens} tokens")
print(f"Total output: {mw.total_response_tokens} tokens")
print(f"Runs:         {mw.run_count}")

# Reset all counters
mw.reset()
```

## Next Steps

- [Cost Tracking Example](../examples/cost-tracking.md) - Full working example
- [Middleware Chains](middleware-chains.md) - Combine cost tracking with other middleware
- [Hook Timeouts](hook-timeouts.md) - Add timeouts to middleware hooks

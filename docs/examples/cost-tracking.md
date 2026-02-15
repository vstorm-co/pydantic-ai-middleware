# Cost Tracking Example

Track token usage and USD costs across agent runs.

## Basic Cost Tracking

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_middleware import MiddlewareAgent, MiddlewareContext
from pydantic_ai_middleware.cost_tracking import create_cost_tracking_middleware


def on_cost(info):
    print(
        f"Run #{info.run_count}: "
        f"{info.run_request_tokens} in / {info.run_response_tokens} out"
    )
    if info.run_cost_usd is not None:
        print(f"  Run cost:   ${info.run_cost_usd:.4f}")
        print(f"  Total cost: ${info.total_cost_usd:.4f}")


cost_mw = create_cost_tracking_middleware(
    model_name="openai:gpt-4.1",
    on_cost_update=on_cost,
)

agent = MiddlewareAgent(
    agent=Agent(model=TestModel()),
    middleware=[cost_mw],
    context=MiddlewareContext(),
)

result = await agent.run("What is the capital of France?")
```

## Budget Enforcement

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_middleware import MiddlewareAgent, MiddlewareContext
from pydantic_ai_middleware.cost_tracking import create_cost_tracking_middleware
from pydantic_ai_middleware.exceptions import BudgetExceededError

cost_mw = create_cost_tracking_middleware(
    model_name="openai:gpt-4.1",
    budget_limit_usd=0.50,
)

agent = MiddlewareAgent(
    agent=Agent(model=TestModel()),
    middleware=[cost_mw],
    context=MiddlewareContext(),
)

prompts = [
    "Summarize this report",
    "Translate to Spanish",
    "Generate test cases",
]

for prompt in prompts:
    try:
        result = await agent.run(prompt)
        print(f"OK: {prompt}")
    except BudgetExceededError as e:
        print(f"Stopped: ${e.cost:.4f} >= ${e.budget:.4f} limit")
        break
```

## Cost Tracking with Logging

```python
import logging
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareAgent,
    MiddlewareContext,
)
from pydantic_ai_middleware.cost_tracking import CostTrackingMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        logger.info(f"Starting run: {prompt[:80]}")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        logger.info(f"Finished run: {output}")
        return output


cost_mw = CostTrackingMiddleware(
    model_name="anthropic:claude-sonnet-4-5-20250929",
    budget_limit_usd=5.0,
    on_cost_update=lambda info: logger.info(
        f"Cost: ${info.total_cost_usd:.4f} "
        f"({info.total_request_tokens} in / {info.total_response_tokens} out)"
    ),
)

agent = MiddlewareAgent(
    agent=Agent(model=TestModel()),
    middleware=[LoggingMiddleware(), cost_mw],
    context=MiddlewareContext(),
)

result = await agent.run("Explain middleware patterns")
```

## Async Callback with Database

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_middleware import MiddlewareAgent, MiddlewareContext
from pydantic_ai_middleware.cost_tracking import (
    CostInfo,
    create_cost_tracking_middleware,
)


async def persist_cost(info: CostInfo) -> None:
    """Save cost data to a database."""
    # Replace with your actual database call
    print(
        f"Saving: run={info.run_count}, "
        f"tokens_in={info.run_request_tokens}, "
        f"tokens_out={info.run_response_tokens}, "
        f"cost={info.run_cost_usd}"
    )


cost_mw = create_cost_tracking_middleware(
    model_name="openai:gpt-4.1",
    on_cost_update=persist_cost,
)

agent = MiddlewareAgent(
    agent=Agent(model=TestModel()),
    middleware=[cost_mw],
    context=MiddlewareContext(),
)

result = await agent.run("Generate a summary")
```

## Resetting Counters

```python
from pydantic_ai_middleware.cost_tracking import CostTrackingMiddleware

cost_mw = CostTrackingMiddleware(model_name="openai:gpt-4.1")

# ... run agent multiple times ...

print(f"Session cost: ${cost_mw.total_cost:.4f}")
print(f"Session runs: {cost_mw.run_count}")

# Start a new billing period
cost_mw.reset()
print(f"After reset: ${cost_mw.total_cost:.4f}")
```

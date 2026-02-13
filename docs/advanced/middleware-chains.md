# Middleware Chains

Middleware chains let you build reusable, ordered pipelines of middleware. Chains can be nested
and passed directly anywhere a middleware instance is accepted.

## Why use a chain

- **Reuse** the same pipeline across multiple agents.
- **Compose** smaller chains into larger ones (chain of chains).
- **Guarantee** ordering in one place instead of duplicating lists.

## Example: reusable input pipeline

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_middleware import AgentMiddleware, MiddlewareAgent, MiddlewareChain


class StripInput(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        return str(prompt).strip()


class LowercaseInput(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        return str(prompt).lower()


input_chain = MiddlewareChain(
    [StripInput(), LowercaseInput()],
    name="InputPipeline",
)

base_agent = Agent(model=TestModel())
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[input_chain],
)

result = await agent.run("  Hello WORLD  ")
```

## Combining chains with `+`

```python
from pydantic_ai_middleware import MiddlewareChain

security = MiddlewareChain([AuthMiddleware(), RateLimitMiddleware()], name="Security")
logging = MiddlewareChain([RequestLog(), ResponseLog()], name="Logging")

# Combine with + operator
full_pipeline = security + logging
```

## Chain of chains

```python
from pydantic_ai_middleware import MiddlewareChain

security = MiddlewareChain([StripInput(), LowercaseInput()], name="Security")
logging = MiddlewareChain([MyLogger()], name="Logging")

pipeline = MiddlewareChain([logging, security], name="FullPipeline")
```

## Execution order

- `before_*` hooks run left-to-right in the chain.
- `after_*` hooks run right-to-left in the chain.

Nested chains are flattened in order, so the guarantees remain predictable.

## Next Steps

- [Parallel Execution](parallel-execution.md) - Run middleware concurrently
- [Conditional Routing](conditional-middleware.md) - Branch based on conditions
- [API Reference](../api/chain.md) - MiddlewareChain API

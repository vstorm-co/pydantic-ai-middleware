# MiddlewareChain

Compose multiple middleware into a reusable, ordered sequence.

`MiddlewareChain` groups middleware together so they can be managed as a single
unit. Chains can be nested, combined with `+`, and modified dynamically.

## Quick Example

```python
from pydantic_ai_middleware import MiddlewareChain, MiddlewareAgent

# Create reusable middleware chains
security_chain = MiddlewareChain([
    AuthMiddleware(),
    RateLimitMiddleware(),
], name="security")

logging_chain = MiddlewareChain([
    RequestLogMiddleware(),
    ResponseLogMiddleware(),
], name="logging")

# Combine chains
full_chain = security_chain + logging_chain

# Or build incrementally
chain = MiddlewareChain(name="my-chain")
chain.add(AuthMiddleware())
chain.add(LoggingMiddleware())

# Use with an agent
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[full_chain],
)
```

## API Reference

::: pydantic_ai_middleware.MiddlewareChain
    options:
      show_source: true
      members:
        - __init__
        - name
        - middleware
        - add
        - insert
        - remove
        - pop
        - replace
        - clear
        - copy
        - before_run
        - after_run
        - before_model_request
        - before_tool_call
        - after_tool_call
        - on_error

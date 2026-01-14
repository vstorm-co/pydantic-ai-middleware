# Concepts

This section covers the core concepts of pydantic-ai-middleware.

## Overview

pydantic-ai-middleware provides a way to intercept and modify agent behavior at various points in the execution lifecycle. This is useful for:

- **Logging** - Track agent activity
- **Security** - Block dangerous inputs or tool calls
- **Rate Limiting** - Control request frequency
- **Metrics** - Collect performance data
- **Transformations** - Modify inputs and outputs
- **Error Handling** - Catch and handle errors
- **Context Sharing** - Share data between hooks with access control

## Core Components

### AgentMiddleware

The base class for all middleware. Override the methods you need:

```python
from pydantic_ai_middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware[MyDeps]):
    async def before_run(self, prompt, deps, ctx):
        # Called before agent runs
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        # Called after agent finishes
        return output
```

### MiddlewareAgent

Wraps an agent and applies middleware:

```python
from pydantic_ai_middleware import MiddlewareAgent

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[middleware1, middleware2]
)
```

### Decorators

Create middleware from simple functions:

```python
from pydantic_ai_middleware import before_run

@before_run
async def log_input(prompt, deps, ctx):
    print(f"Input: {prompt}")
    return prompt
```

## Learn More

- [Middleware](middleware.md) - Deep dive into middleware
- [Hooks](hooks.md) - All available lifecycle hooks
- [Decorators](decorators.md) - Decorator-based middleware
- [Parallel Execution](parallel-execution.md) - Run middleware in parallel
- [Conditional Middleware](conditional-middleware.md) - Branch execution based on conditions

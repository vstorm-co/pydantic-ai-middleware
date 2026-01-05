# pydantic-ai-middleware

Simple middleware library for [Pydantic-AI](https://github.com/pydantic/pydantic-ai) - clean before/after hooks without imposed guardrails structure.

## Why pydantic-ai-middleware?

Pydantic-AI is a powerful framework for building AI agents. However, when you need to add cross-cutting concerns like logging, security checks, rate limiting, or metrics, you often end up with boilerplate code scattered throughout your application.

**pydantic-ai-middleware** provides a clean, composable way to intercept and modify agent behavior at various points in the execution lifecycle.

## Key Features

- **Clean Middleware API** - Simple before/after hooks at every lifecycle stage
- **No Imposed Structure** - You decide what to do (logging, guardrails, metrics, transformations)
- **Full Control** - Modify prompts, outputs, tool calls, and handle errors
- **Context Sharing** - Share data between middleware with access control
- **Decorator Support** - Simple decorators for quick middleware creation
- **Type Safe** - Full typing support with generics for dependencies

## Quick Example

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware, MiddlewareContext

class LoggingMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        print(f"Starting: {prompt}")
        if ctx:
            ctx.set("logged", True)  # Store data for later hooks
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        print(f"Finished: {output}")
        return output

# Create context with config (optional)
ctx = MiddlewareContext(config={"log_level": "debug"})

# Wrap your agent with middleware
agent = MiddlewareAgent(
    agent=Agent('openai:gpt-4o'),
    middleware=[LoggingMiddleware()],
    context=ctx,
)

result = await agent.run("Hello!")
```

## Installation

```bash
pip install pydantic-ai-middleware
```

## Next Steps

- [Installation](installation.md) - Detailed installation instructions
- [Quick Start](quickstart.md) - Get started in minutes
- [Concepts](concepts/index.md) - Understand how middleware works
- [Examples](examples/index.md) - Real-world examples

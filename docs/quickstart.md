# Quick Start

This guide will get you up and running with pydantic-ai-middleware in minutes.

## Basic Usage

### 1. Create a Middleware Class

```python
from pydantic_ai_middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware[None]):
    """Log agent activity."""

    async def before_run(self, prompt, deps, ctx):
        print(f"Starting with prompt: {prompt}")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        print(f"Finished with output: {output}")
        return output
```

### 2. Wrap Your Agent

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent

# Create your base agent
base_agent = Agent('openai:gpt-4o')

# Wrap with middleware
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[LoggingMiddleware()],
)
```

### 3. Use Normally

```python
result = await agent.run("What is 2 + 2?")
print(result.output)
```

## Using Decorators

For simple middleware, use decorators:

```python
from pydantic_ai_middleware import before_run, after_run

@before_run
async def log_input(prompt, deps, ctx):
    print(f"Input: {prompt}")
    return prompt

@after_run
async def log_output(prompt, output, deps, ctx):
    print(f"Output: {output}")
    return output

# Use as middleware
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[log_input, log_output],
)
```

## Blocking Inputs

Block unwanted inputs by raising `InputBlocked`:

```python
from pydantic_ai_middleware import AgentMiddleware, InputBlocked

class ContentFilter(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        if "forbidden" in prompt.lower():
            raise InputBlocked("Content not allowed")
        return prompt
```

## Blocking Tools

Block specific tool calls with `ToolBlocked`:

```python
from pydantic_ai_middleware import AgentMiddleware, ToolBlocked

class ToolFilter(AgentMiddleware[None]):
    blocked_tools = {"dangerous_tool", "admin_tool"}

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        if tool_name in self.blocked_tools:
            raise ToolBlocked(tool_name, "Tool not allowed")
        return tool_args
```

## Multiple Middleware

Combine multiple middleware:

```python
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[
        RateLimitMiddleware(),
        LoggingMiddleware(),
        SecurityMiddleware(),
    ],
)
```

Middleware executes in order for `before_*` hooks and reverse order for `after_*` hooks.

## Next Steps

- Learn about [middleware concepts](concepts/middleware.md)
- See [real-world examples](examples/index.md)
- Explore the [API reference](api/index.md)

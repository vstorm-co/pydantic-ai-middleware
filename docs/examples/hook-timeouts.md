# Hook Timeouts Example

Prevent slow middleware from blocking agent execution.

## Basic Timeout

```python
from pydantic_ai_middleware import AgentMiddleware, MiddlewareTimeout

class ExternalAPICheck(AgentMiddleware[None]):
    """Check input against an external API with a 3-second timeout."""
    timeout = 3.0

    async def before_run(self, prompt, deps, ctx):
        result = await call_moderation_api(prompt)
        if result.flagged:
            raise InputBlocked("Content flagged by moderation API")
        return prompt
```

## Handling Timeouts

```python
from pydantic_ai_middleware import MiddlewareAgent, MiddlewareTimeout

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[ExternalAPICheck()],
)

try:
    result = await agent.run("test input")
except MiddlewareTimeout as e:
    print(f"Middleware '{e.middleware_name}' timed out")
    print(f"Hook: {e.hook_name}, Timeout: {e.timeout}s")
    # Fall back to running without the slow middleware
```

## Timeout with Tool Filtering

```python
class SlowToolGuard(AgentMiddleware[None]):
    """Guard specific tools with a timeout."""
    tool_names = {"web_search"}
    timeout = 5.0

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        await validate_search_query(tool_args["query"])
        return tool_args

    async def on_tool_error(self, tool_name, tool_args, error, deps, ctx):
        await log_search_error(tool_name, error)
        return None
```

Source: [`examples/hook_timeouts.py`](https://github.com/vstorm-co/pydantic-ai-middleware/blob/main/examples/hook_timeouts.py)

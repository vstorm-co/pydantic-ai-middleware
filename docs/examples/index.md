# Examples

Real-world examples of using pydantic-ai-middleware.

## Quick Links

- [Logging](logging.md) - Log agent activity
- [Security](security.md) - Implement security checks
- [Rate Limiting](rate-limiting.md) - Control request frequency

## Script Examples

- `examples/conditional_middleware.py` - Conditional middleware driven by context config

## Basic Example

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware

class SimpleLogger(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        print(f">> {prompt}")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        print(f"<< {output}")
        return output

agent = MiddlewareAgent(
    agent=Agent('openai:gpt-4o'),
    middleware=[SimpleLogger()],
)

result = await agent.run("Hello!")
# >> Hello!
# << Hi there! How can I help you?
```

## Combining Middleware

```python
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[
        RateLimitMiddleware(max_calls=10, window=60),
        LoggingMiddleware(),
        SecurityMiddleware(),
        MetricsMiddleware(),
    ],
)
```

Each middleware handles a specific concern, keeping your code clean and modular.

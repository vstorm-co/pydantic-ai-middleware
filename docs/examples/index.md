# Examples

Real-world examples of using pydantic-ai-middleware.

## Getting Started

| Example | Description |
|---------|-------------|
| [Logging](logging.md) | Log agent activity with before/after hooks |
| [Rate Limiting](rate-limiting.md) | Control request frequency with sliding window |

## Security

| Example | Description |
|---------|-------------|
| [Security](security.md) | Input validation, tool authorization, PII redaction |
| [PII Detection](pii-detection.md) | Detect and redact personally identifiable information |
| [Prompt Injection](prompt-injection.md) | Screen prompts for injection techniques |
| [Toxicity Check](toxicity-check.md) | Screen content for toxic or offensive language |
| [JSON Schema Validation](json-schema-validation.md) | Validate tool arguments against JSON schemas |

## Quick Example

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

## Runnable Examples

The `examples/` directory contains standalone scripts you can run directly:

```bash
uv run python examples/pii_detection.py
uv run python examples/prompt_injection.py
uv run python examples/toxicity_check.py
uv run python examples/json_schema_validation.py
```

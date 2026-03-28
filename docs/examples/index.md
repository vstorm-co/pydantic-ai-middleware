# Examples

Real-world examples of using pydantic-ai-shields.

## Security

| Example | Description |
|---------|-------------|
| [Security Overview](security.md) | Combining multiple shields for defense-in-depth |
| [PII Detection](pii-detection.md) | Detect personally identifiable information |
| [Prompt Injection](prompt-injection.md) | Block injection and jailbreak attempts |
| [Toxicity Check](toxicity-check.md) | Screen content for toxic language |
| [JSON Schema Validation](json-schema-validation.md) | Validate tool arguments |

## Cost Control

| Example | Description |
|---------|-------------|
| [Cost Tracking](cost-tracking.md) | Token usage monitoring and budget enforcement |

## Quick Example

```python
from pydantic_ai import Agent
from pydantic_ai_shields import (
    CostTracking, PromptInjection, PiiDetector, SecretRedaction, NoRefusals,
)

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        CostTracking(budget_usd=5.0),
        PromptInjection(sensitivity="high"),
        PiiDetector(),
        SecretRedaction(),
        NoRefusals(),
    ],
)

result = await agent.run("Hello!")
```

Each shield handles a specific concern, keeping your code clean and modular.

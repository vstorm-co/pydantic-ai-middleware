# Security

Combine multiple shields for defense-in-depth:

```python
from pydantic_ai import Agent
from pydantic_ai_shields import (
    PromptInjection, PiiDetector, SecretRedaction,
    BlockedKeywords, ToolGuard, NoRefusals,
)

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        # Input shields
        PromptInjection(sensitivity="high"),
        PiiDetector(detect=["email", "ssn", "credit_card"]),
        BlockedKeywords(keywords=["internal_only", "classified"]),

        # Tool shield
        ToolGuard(blocked=["execute"], require_approval=["write_file"]),

        # Output shields
        SecretRedaction(),
        NoRefusals(),
    ],
)
```

Each shield fires at the appropriate lifecycle point — input shields check the prompt before the model runs, output shields check the response after.

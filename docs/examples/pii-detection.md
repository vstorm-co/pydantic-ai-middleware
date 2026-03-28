# PII Detection

Detect personally identifiable information in user input.

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_shields import PiiDetector

agent = Agent("openai:gpt-4.1", capabilities=[PiiDetector()])

# This will raise InputBlocked:
await agent.run("Email me at john@example.com")
```

## Detect Specific Types

```python
agent = Agent("openai:gpt-4.1", capabilities=[
    PiiDetector(detect=["email", "ssn", "credit_card"]),
])
```

Built-in types: `email`, `phone`, `ssn`, `credit_card`, `ip_address`.

## Log Instead of Block

```python
detector = PiiDetector(action="log")
agent = Agent("openai:gpt-4.1", capabilities=[detector])

result = await agent.run("My email is test@example.com")
# Allowed through, but recorded:
print(detector.last_detections)
# [{"type": "email", "count": 1}]
```

## Custom PII Patterns

```python
agent = Agent("openai:gpt-4.1", capabilities=[
    PiiDetector(custom_patterns={
        "passport": r"[A-Z]{2}\d{7}",
        "nhs_number": r"\d{3}\s\d{3}\s\d{4}",
    }),
])
```

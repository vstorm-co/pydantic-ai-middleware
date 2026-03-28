# Toxicity Check

Screen input for toxic content using `BlockedKeywords` or a custom `InputGuard`:

## With BlockedKeywords

```python
from pydantic_ai import Agent
from pydantic_ai_shields import BlockedKeywords

agent = Agent("openai:gpt-4.1", capabilities=[
    BlockedKeywords(keywords=["offensive_term1", "offensive_term2"]),
])
```

## With Custom InputGuard

For more sophisticated detection (e.g., calling a moderation API):

```python
from pydantic_ai import Agent
from pydantic_ai_shields import InputGuard

async def check_toxicity(prompt: str) -> bool:
    # Call your moderation API here
    result = await moderation_api.check(prompt)
    return result.is_safe

agent = Agent("openai:gpt-4.1", capabilities=[InputGuard(guard=check_toxicity)])
```

## Combined Input + Output Check

```python
from pydantic_ai_shields import InputGuard, OutputGuard

agent = Agent("openai:gpt-4.1", capabilities=[
    InputGuard(guard=check_toxicity),
    OutputGuard(guard=check_toxicity),
])
```

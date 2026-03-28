# JSON Schema Validation

Validate model output format using `OutputGuard`:

```python
import json
from pydantic_ai import Agent
from pydantic_ai_shields import OutputGuard

def is_valid_json(output: str) -> bool:
    """Ensure output is valid JSON."""
    try:
        json.loads(output)
        return True
    except json.JSONDecodeError:
        return False

agent = Agent("openai:gpt-4.1", capabilities=[OutputGuard(guard=is_valid_json)])
```

## Validate Against Schema

```python
import json
from pydantic_ai_shields import OutputGuard

def validate_response_schema(output: str) -> bool:
    """Ensure output has required fields."""
    try:
        data = json.loads(output)
        return "status" in data and "result" in data
    except (json.JSONDecodeError, TypeError):
        return False

agent = Agent("openai:gpt-4.1", capabilities=[OutputGuard(guard=validate_response_schema)])
```

!!! tip
    For structured output, consider using pydantic-ai's built-in `output_type` parameter
    with a Pydantic model — it validates output at the framework level.

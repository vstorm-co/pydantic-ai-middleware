# Exceptions

All shield exceptions inherit from `GuardrailError`.

## GuardrailError

Base exception for all shield violations.

## InputBlocked

Raised when input fails validation. Thrown by `InputGuard`, `PromptInjection`, `PiiDetector`, `BlockedKeywords`.

```python
from pydantic_ai_shields import InputBlocked

try:
    result = await agent.run("bad input")
except InputBlocked as e:
    print(f"Blocked: {e}")
```

## OutputBlocked

Raised when output fails validation. Thrown by `OutputGuard`, `SecretRedaction`, `NoRefusals`.

```python
from pydantic_ai_shields import OutputBlocked

try:
    result = await agent.run("prompt")
except OutputBlocked as e:
    print(f"Output blocked: {e}")
```

## ToolBlocked

Raised when a tool call is denied. Thrown by `ToolGuard`.

```python
from pydantic_ai_shields import ToolBlocked

try:
    result = await agent.run("execute rm -rf /")
except ToolBlocked as e:
    print(f"Tool '{e.tool_name}' blocked: {e.reason}")
```

## BudgetExceededError

Raised when cumulative cost exceeds the budget. Thrown by `CostTracking`.

```python
from pydantic_ai_shields import BudgetExceededError

try:
    result = await agent.run("expensive query")
except BudgetExceededError as e:
    print(f"Over budget: ${e.total_cost:.4f} > ${e.budget:.4f}")
```

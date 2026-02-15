# Exceptions

Custom exceptions for middleware control flow and error reporting. All exceptions
inherit from `MiddlewareError`, making it easy to catch any middleware-related error
with a single `except` clause.

The blocking exceptions (`InputBlocked`, `ToolBlocked`, `OutputBlocked`) are the
primary way middleware communicates that a request should be rejected. Raise them from
any hook to stop processing immediately.

## MiddlewareError

::: pydantic_ai_middleware.MiddlewareError

Base exception for all middleware errors.

## MiddlewareConfigError

::: pydantic_ai_middleware.exceptions.MiddlewareConfigError

Raised when middleware configuration is invalid (unknown type names, bad predicate
specs, malformed config files).

## InputBlocked

::: pydantic_ai_middleware.InputBlocked

Raised to block input processing.

```python
from pydantic_ai_middleware import InputBlocked

raise InputBlocked("Content not allowed")
raise InputBlocked()  # Uses default message
```

## ToolBlocked

::: pydantic_ai_middleware.ToolBlocked

Raised to block a tool call.

```python
from pydantic_ai_middleware import ToolBlocked

raise ToolBlocked("dangerous_tool", "Not authorized")
raise ToolBlocked("tool_name")  # Uses default reason
```

## OutputBlocked

::: pydantic_ai_middleware.OutputBlocked

Raised to block output.

```python
from pydantic_ai_middleware import OutputBlocked

raise OutputBlocked("Contains sensitive information")
raise OutputBlocked()  # Uses default message
```

## BudgetExceededError

::: pydantic_ai_middleware.exceptions.BudgetExceededError

Raised when accumulated cost exceeds the configured budget limit. Used by
`CostTrackingMiddleware`.

```python
from pydantic_ai_middleware.exceptions import BudgetExceededError

# Raised automatically by CostTrackingMiddleware
# You can also catch it:
try:
    result = await agent.run("prompt")
except BudgetExceededError as e:
    print(f"Over budget: ${e.cost:.4f} >= ${e.budget:.4f}")
```

## ParallelExecutionFailed

::: pydantic_ai_middleware.exceptions.ParallelExecutionFailed

Raised when parallel middleware execution fails. Contains the list of errors and any
successful results.

## GuardrailTimeout

::: pydantic_ai_middleware.exceptions.GuardrailTimeout

Raised when an async guardrail exceeds its configured timeout.

## MiddlewareTimeout

::: pydantic_ai_middleware.exceptions.MiddlewareTimeout

Raised when a middleware hook exceeds its per-middleware timeout.

## AggregationFailed

::: pydantic_ai_middleware.exceptions.AggregationFailed

Raised when parallel results cannot be aggregated according to the chosen strategy.

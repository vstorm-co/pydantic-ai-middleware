# Conditional Middleware

Conditional middleware allows you to route execution to different middleware based on runtime conditions.

## Basic Usage

Use `ConditionalMiddleware` to branch based on a predicate function:

```python
from pydantic_ai_middleware import ConditionalMiddleware

def is_admin(ctx):
    return ctx is not None and ctx.config.get("role") == "admin"

middleware = ConditionalMiddleware(
    condition=is_admin,
    when_true=AdminMiddleware(),
    when_false=UserMiddleware(),
)
```

## Predicate Functions

Predicates receive a `ScopedContext` and return a boolean:

```python
from pydantic_ai_middleware import ScopedContext, HookType

def is_after_run(ctx: ScopedContext | None) -> bool:
    return ctx is not None and ctx.current_hook == HookType.AFTER_RUN

def has_feature_flag(ctx: ScopedContext | None) -> bool:
    return ctx is not None and ctx.config.get("feature_enabled", False)
```

## Middleware Pipelines

Both branches can accept a single middleware or a sequence:

```python
middleware = ConditionalMiddleware(
    condition=is_admin,
    when_true=[AuditMiddleware(), AdminMiddleware()],  # Pipeline
    when_false=UserMiddleware(),  # Single middleware
)
```

## Optional Else Branch

The `when_false` branch is optional. When omitted, the condition acts as a guard:

```python
middleware = ConditionalMiddleware(
    condition=lambda ctx: ctx is not None and ctx.config.get("logging_enabled"),
    when_true=LoggingMiddleware(),
    # No when_false - just pass through when condition is False
)
```

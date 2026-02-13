# ConditionalMiddleware

Route to different middleware based on runtime conditions.

`ConditionalMiddleware` evaluates a predicate function at each hook invocation
to decide which middleware pipeline to execute. This enables dynamic branching
in your middleware chain based on context, user state, or other runtime factors.

## Quick Example

```python
from pydantic_ai_middleware import ConditionalMiddleware
from pydantic_ai_middleware.context import HookType, ScopedContext

def is_tool_hook(ctx: ScopedContext | None) -> bool:
    """Route tool-related hooks to specialized middleware."""
    if ctx is None:
        return False
    return ctx.current_hook in (HookType.BEFORE_TOOL_CALL, HookType.AFTER_TOOL_CALL)

middleware = ConditionalMiddleware(
    condition=is_tool_hook,
    when_true=ToolAuditMiddleware(),
    when_false=GeneralAuditMiddleware(),
)
```

## API Reference

::: pydantic_ai_middleware.ConditionalMiddleware
    options:
      show_source: true
      members:
        - __init__
        - before_run
        - after_run
        - before_model_request
        - before_tool_call
        - after_tool_call
        - on_error

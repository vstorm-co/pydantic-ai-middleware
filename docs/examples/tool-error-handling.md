# Tool Error Handling Example

Handle tool failures with context-aware error recovery.

## Retry with Fallback

```python
from pydantic_ai_middleware import AgentMiddleware, on_tool_error

class APIErrorHandler(AgentMiddleware[None]):
    """Convert API errors into user-friendly messages."""
    tool_names = {"web_search", "api_call"}

    async def on_tool_error(self, tool_name, tool_args, error, deps, ctx):
        if isinstance(error, TimeoutError):
            return ConnectionError(
                f"{tool_name} timed out. Query: {tool_args.get('query', 'unknown')}"
            )
        if isinstance(error, PermissionError):
            return RuntimeError(f"Access denied for {tool_name}")
        return None  # re-raise original for unhandled errors
```

## Using Decorators

```python
@on_tool_error(tools={"send_email"})
async def email_error_handler(tool_name, tool_args, error, deps, ctx):
    recipient = tool_args.get("to", "unknown")
    return RuntimeError(f"Failed to email {recipient}: {error}")

@on_tool_error
async def global_error_logger(tool_name, tool_args, error, deps, ctx):
    print(f"[ERROR] {tool_name}({tool_args}): {error}")
    return None  # always re-raise, just log
```

## Error Context Tracking

```python
class ErrorTracker(AgentMiddleware[None]):
    """Track tool errors in context for reporting."""

    async def on_tool_error(self, tool_name, tool_args, error, deps, ctx):
        if ctx:
            errors = ctx.metadata.get("tool_errors", [])
            errors.append({
                "tool": tool_name,
                "error": str(error),
                "args": tool_args,
            })
            ctx.metadata["tool_errors"] = errors
        return None  # re-raise
```

Source: [`examples/tool_error_handling.py`](https://github.com/vstorm-co/pydantic-ai-middleware/blob/main/examples/tool_error_handling.py)

# Tool Error Handling

The `on_tool_error` hook lets you handle tool execution failures with full tool context.

## The Problem

Before `on_tool_error`, tool failures went through the generic `on_error` handler, losing the tool name and arguments:

```python
async def on_error(self, error, deps, ctx):
    # Which tool failed? What arguments were passed?
    # You don't know.
    return None
```

## The Solution

```python
class RetryableToolHandler(AgentMiddleware[None]):
    async def on_tool_error(self, tool_name, tool_args, error, deps, ctx):
        if tool_name == "web_search" and isinstance(error, TimeoutError):
            return ConnectionError(f"Search timed out for: {tool_args.get('query')}")
        return None  # re-raise original
```

## Signature

```python
async def on_tool_error(
    self,
    tool_name: str,
    tool_args: dict[str, Any],
    error: Exception,
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> Exception | None:
```

**Return values:**

- `None` -- re-raise the original exception
- `Exception` -- raise this exception instead

## Execution Order

`on_tool_error` fires **between** `before_tool_call` and `after_tool_call`:

```
before_tool_call -> tool execution -> on_tool_error (if error) -> after_tool_call (if success)
```

If `on_tool_error` returns a replacement exception, that exception is raised and `after_tool_call` is skipped.

## Tool Name Filtering

Combine with `tool_names` for scoped error handling:

```python
class SearchErrorHandler(AgentMiddleware[None]):
    tool_names = {"web_search", "api_call"}

    async def on_tool_error(self, tool_name, tool_args, error, deps, ctx):
        if isinstance(error, TimeoutError):
            return ConnectionError(f"{tool_name} timed out")
        return None
```

## Decorator Syntax

```python
from pydantic_ai_middleware import on_tool_error

@on_tool_error
async def log_all_errors(tool_name, tool_args, error, deps, ctx):
    print(f"Tool {tool_name} failed: {error}")
    return None  # re-raise

@on_tool_error(tools={"send_email"})
async def handle_email_errors(tool_name, tool_args, error, deps, ctx):
    return RuntimeError(f"Email failed: {error}")
```

## In Composite Middleware

`on_tool_error` is supported in all composite middleware:

- **MiddlewareChain** -- first handler to return non-`None` wins
- **ParallelMiddleware** -- runs handlers concurrently
- **ConditionalMiddleware** -- routes to selected branch
- **AsyncGuardrailMiddleware** -- delegates to wrapped guardrail (BLOCKING mode only)

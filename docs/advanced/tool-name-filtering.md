# Tool Name Filtering

Scope middleware to specific tools instead of handling every tool call.

## The Problem

Without filtering, every `before_tool_call` / `after_tool_call` / `on_tool_error` middleware fires for every tool. You end up with manual checks:

```python
class EmailGuard(AgentMiddleware[None]):
    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        if tool_name not in ("send_email", "draft_email"):
            return tool_args  # skip
        # actual logic...
```

## The Solution

Set `tool_names` on the middleware class:

```python
class EmailGuard(AgentMiddleware[None]):
    tool_names = {"send_email", "draft_email"}

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        # Only called for send_email and draft_email
        if not tool_args.get("to"):
            raise ToolBlocked(tool_name, "Recipient required")
        return tool_args
```

## How It Works

- `tool_names = None` (default) -- middleware handles **all** tools
- `tool_names = {"tool_a", "tool_b"}` -- middleware handles only matching tools
- `tool_names = set()` -- middleware handles **no** tools (effectively disabled)

The `_should_handle_tool(tool_name)` method performs the check. Filtering applies to:

- `before_tool_call`
- `on_tool_error`
- `after_tool_call`

Other hooks (`before_run`, `after_run`, `before_model_request`, `on_error`) are not affected.

## Filtering in Composite Middleware

Tool name filtering works in all composite middleware types:

### MiddlewareChain

```python
chain = MiddlewareChain([
    EmailGuard(),        # tool_names = {"send_email"}
    LoggingMiddleware(), # tool_names = None (all tools)
])

# send_email -> both fire
# read_file  -> only LoggingMiddleware fires
```

### MiddlewareToolset

```python
toolset = MiddlewareToolset(
    wrapped=base_toolset,
    middleware=[EmailGuard(), FileGuard()],
)
# Each middleware only fires for matching tools
```

## Decorator Syntax

Use the `tools` parameter on decorators:

```python
from pydantic_ai_middleware import before_tool_call, after_tool_call, on_tool_error

@before_tool_call(tools={"send_email"})
async def validate_email(tool_name, tool_args, deps, ctx):
    return tool_args

@after_tool_call(tools={"read_file"})
async def log_read(tool_name, tool_args, result, deps, ctx):
    print(f"Read: {tool_args}")
    return result

@on_tool_error(tools={"web_search"})
async def handle_search_error(tool_name, tool_args, error, deps, ctx):
    return ConnectionError("Search unavailable")
```

Plain decorators (without `tools`) match all tools:

```python
@before_tool_call
async def validate_all(tool_name, tool_args, deps, ctx):
    return tool_args
```

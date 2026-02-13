# Tool Name Filtering Example

Scope middleware to specific tools for focused validation.

## Email-Only Validation

```python
from pydantic_ai_middleware import AgentMiddleware, ToolBlocked

class EmailValidator(AgentMiddleware[None]):
    """Only validates send_email and draft_email calls."""
    tool_names = {"send_email", "draft_email"}

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        if not tool_args.get("to"):
            raise ToolBlocked(tool_name, "Recipient email is required")
        if not tool_args.get("subject"):
            raise ToolBlocked(tool_name, "Subject is required")
        return tool_args

    async def after_tool_call(self, tool_name, tool_args, result, deps, ctx):
        print(f"Email sent to {tool_args['to']}: {tool_args['subject']}")
        return result
```

## Using Decorators

```python
from pydantic_ai_middleware import before_tool_call, after_tool_call

@before_tool_call(tools={"execute_code"})
async def sandbox_check(tool_name, tool_args, deps, ctx):
    if "rm -rf" in tool_args.get("code", ""):
        raise ToolBlocked(tool_name, "Dangerous code detected")
    return tool_args

@after_tool_call(tools={"web_search"})
async def cache_results(tool_name, tool_args, result, deps, ctx):
    if ctx:
        ctx.set("last_search", result)
    return result
```

## Mixed Filtering in a Chain

```python
from pydantic_ai_middleware import MiddlewareChain

chain = MiddlewareChain([
    EmailValidator(),    # only send_email, draft_email
    sandbox_check,       # only execute_code
    cache_results,       # only web_search
    LoggingMiddleware(), # all tools (tool_names = None)
])
```

Source: [`examples/tool_name_filtering.py`](https://github.com/vstorm-co/pydantic-ai-middleware/blob/main/examples/tool_name_filtering.py)

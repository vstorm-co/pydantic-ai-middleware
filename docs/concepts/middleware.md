# Middleware

Middleware is the core concept of pydantic-ai-middleware. It allows you to intercept and modify agent behavior at various points in the execution lifecycle.

## Creating Middleware

Create middleware by inheriting from `AgentMiddleware`:

```python
from pydantic_ai_middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        # Your logic here
        return prompt
```

## Generic Dependencies

Middleware supports typed dependencies:

```python
from dataclasses import dataclass

@dataclass
class MyDeps:
    user_id: str
    is_admin: bool

class AuthMiddleware(AgentMiddleware[MyDeps]):
    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx):
        if tool_name == "admin_tool" and not deps.is_admin:
            raise ToolBlocked(tool_name, "Admin only")
        return tool_args
```

## Middleware Execution Order

Middleware executes in a specific order:

- **before_*** hooks execute in the order middleware is listed
- **after_*** hooks execute in reverse order

```python
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[mw1, mw2, mw3],
)

# Execution order:
# before_run: mw1 -> mw2 -> mw3 -> [agent runs]
# after_run:  [agent done] -> mw3 -> mw2 -> mw1
```

This is similar to how ASGI/WSGI middleware works.

## Modifying Data

Middleware can modify data at each stage:

```python
class TransformMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        # Modify the prompt
        return f"[Context] {prompt}"

    async def after_run(self, prompt, output, deps, ctx):
        # Modify the output
        return {"original": output, "processed": True}
```

## Blocking Execution

Block execution by raising exceptions:

```python
from pydantic_ai_middleware import InputBlocked, ToolBlocked

class SecurityMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        if is_malicious(prompt):
            raise InputBlocked("Malicious content detected")
        return prompt

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        if tool_name in BLOCKED_TOOLS:
            raise ToolBlocked(tool_name, "Tool not allowed")
        return tool_args
```

## Error Handling

Handle errors with the `on_error` hook:

```python
class ErrorHandler(AgentMiddleware[None]):
    async def on_error(self, error, deps, ctx):
        # Log the error
        logger.error(f"Agent error: {error}")

        # Return None to re-raise the original
        return None

        # Or return a different exception
        # return UserFriendlyError("Something went wrong")
```

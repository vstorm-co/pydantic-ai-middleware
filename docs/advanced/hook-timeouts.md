# Hook Timeouts

Set per-middleware timeouts to prevent slow hooks from blocking the agent indefinitely.

## Basic Usage

```python
class SlowGuard(AgentMiddleware[None]):
    timeout = 5.0  # seconds

    async def before_run(self, prompt, deps, ctx):
        result = await external_api_check(prompt)  # if >5s -> MiddlewareTimeout
        return prompt
```

If any hook on this middleware exceeds 5 seconds, a `MiddlewareTimeout` exception is raised.

## How It Works

- `timeout = None` (default) -- no timeout, hooks can run indefinitely
- `timeout = 5.0` -- all hooks on this middleware are wrapped with `asyncio.wait_for(coro, timeout=5.0)`
- The timeout applies to **every hook** on the middleware: `before_run`, `after_run`, `before_tool_call`, `on_tool_error`, `after_tool_call`, `on_error`, etc.

## MiddlewareTimeout Exception

```python
from pydantic_ai_middleware import MiddlewareTimeout

try:
    result = await agent.run("test")
except MiddlewareTimeout as e:
    print(e.middleware_name)  # "SlowGuard"
    print(e.timeout)          # 5.0
    print(e.hook_name)        # "before_run"
```

## Where Timeouts Are Enforced

Timeouts are enforced at call sites:

- **MiddlewareToolset** -- wraps `before_tool_call`, `on_tool_error`, `after_tool_call`
- **MiddlewareAgent** -- wraps `before_run`, `after_run`, `on_error`

## Combining with Other Features

### Timeout + Tool Name Filtering

```python
class EmailTimeout(AgentMiddleware[None]):
    tool_names = {"send_email"}
    timeout = 10.0

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        await validate_recipient(tool_args["to"])
        return tool_args
```

### Timeout + on_tool_error

```python
class ErrorRecovery(AgentMiddleware[None]):
    timeout = 3.0

    async def on_tool_error(self, tool_name, tool_args, error, deps, ctx):
        await log_error(tool_name, error)  # if >3s -> MiddlewareTimeout
        return None
```

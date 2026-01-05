# Lifecycle Hooks

pydantic-ai-middleware provides hooks at various points in the agent execution lifecycle.

## Available Hooks

| Hook | When Called | Can Modify | Can Block |
|------|-------------|------------|-----------|
| `before_run` | Before agent starts | Prompt | Yes (`InputBlocked`) |
| `after_run` | After agent finishes | Output | Yes (`OutputBlocked`) |
| `before_model_request` | Before each model call | Messages | No |
| `before_tool_call` | Before tool execution | Tool arguments | Yes (`ToolBlocked`) |
| `after_tool_call` | After tool execution | Tool result | No |
| `on_error` | When error occurs | Exception | Can convert |

## Hook Execution Order

Hooks execute in a specific order, which matters for context sharing:

1. **BEFORE_RUN** (1) - Initial input processing
2. **BEFORE_MODEL_REQUEST** (2) - Before sending to model
3. **BEFORE_TOOL_CALL** (3) - Before tool execution
4. **AFTER_TOOL_CALL** (4) - After tool execution
5. **AFTER_RUN** (5) - Final output processing
6. **ON_ERROR** (6) - Error handling (can read all hooks)

## Context Parameter

All hooks receive an optional `ctx` parameter for sharing data:

```python
from pydantic_ai_middleware import ScopedContext

async def before_run(
    self,
    prompt: str | Sequence[Any],
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> str | Sequence[Any]:
    if ctx:
        ctx.set("start_time", time.time())  # Store data
    return prompt
```

## before_run

Called before the agent starts processing. Can modify the prompt or block execution.

```python
async def before_run(
    self,
    prompt: str | Sequence[Any],
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> str | Sequence[Any]:
    # Return modified prompt
    return prompt
```

## after_run

Called after the agent finishes. Can modify the output.

```python
async def after_run(
    self,
    prompt: str | Sequence[Any],
    output: Any,
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> Any:
    # Return modified output
    return output
```

## before_model_request

Called before each request to the model. Can modify messages.

```python
async def before_model_request(
    self,
    messages: list[ModelMessage],
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> list[ModelMessage]:
    # Return modified messages
    return messages
```

## before_tool_call

Called before a tool is executed. Can modify arguments or block.

```python
async def before_tool_call(
    self,
    tool_name: str,
    tool_args: dict[str, Any],
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> dict[str, Any]:
    # Return modified arguments
    return tool_args
```

## after_tool_call

Called after a tool is executed. Can modify the result.

```python
async def after_tool_call(
    self,
    tool_name: str,
    tool_args: dict[str, Any],
    result: Any,
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> Any:
    # Return modified result
    return result
```

## on_error

Called when an error occurs. Can log, transform, or re-raise.

```python
async def on_error(
    self,
    error: Exception,
    deps: DepsT | None,
    ctx: ScopedContext | None = None,
) -> Exception | None:
    # Return None to re-raise original
    # Return exception to raise different one
    return None
```

## Context Sharing

Middleware can share data through the context system. Each hook has its own namespace, and hooks can read from earlier hooks in the execution chain.

### Storing Data

```python
async def before_run(self, prompt, deps, ctx: ScopedContext | None = None):
    if ctx:
        ctx.set("user_id", "123")  # Stored in BEFORE_RUN namespace
        ctx.set("timestamp", time.time())
    return prompt
```

### Reading Data from Earlier Hooks

```python
from pydantic_ai_middleware import HookType

async def after_run(self, prompt, output, deps, ctx: ScopedContext | None = None):
    if ctx:
        # Read from BEFORE_RUN namespace
        user_id = ctx.get_from(HookType.BEFORE_RUN, "user_id")
        timestamp = ctx.get_from(HookType.BEFORE_RUN, "timestamp")
        
        elapsed = time.time() - timestamp
        print(f"User {user_id} request took {elapsed:.2f}s")
    return output
```

### Access Control Rules

- Hooks can only **write** to their own namespace
- Hooks can **read** from earlier hooks (lower number in execution order)
- Hooks **cannot read** from later hooks
- `ON_ERROR` can read from **all** hooks

```python
# ✓ AFTER_RUN can read from BEFORE_RUN
after_ctx.get_from(HookType.BEFORE_RUN, "key")

# ✗ BEFORE_RUN cannot read from AFTER_RUN (raises ContextAccessError)
before_ctx.get_from(HookType.AFTER_RUN, "key")
```

### Accessing Global Config

```python
async def before_run(self, prompt, deps, ctx: ScopedContext | None = None):
    if ctx:
        timeout = ctx.config.get("timeout", 30)
        debug = ctx.config.get("debug", False)
        if debug:
            print(f"Processing with timeout={timeout}")
    return prompt

# Config is set when creating the context
ctx = MiddlewareContext(config={"timeout": 60, "debug": True})

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[...],
    context=ctx,
)
```

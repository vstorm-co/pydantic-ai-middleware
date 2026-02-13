# Decorators

For simple middleware, use decorator functions instead of classes.

## Available Decorators

- `@before_run` - Create middleware from a before_run function
- `@after_run` - Create middleware from an after_run function
- `@before_model_request` - Create middleware from a before_model_request function
- `@before_tool_call` - Create middleware from a before_tool_call function
- `@after_tool_call` - Create middleware from an after_tool_call function
- `@on_error` - Create middleware from an on_error function

## Usage

```python
from pydantic_ai_middleware import before_run, after_run

@before_run
async def log_input(prompt, deps, ctx):
    print(f"Input: {prompt}")
    return prompt

@after_run
async def log_output(prompt, output, deps, ctx):
    print(f"Output: {output}")
    return output

# Use as middleware
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[log_input, log_output],
)
```

## With Type Hints

```python
from collections.abc import Sequence
from typing import Any
from pydantic_ai_middleware.context import ScopedContext

@before_run
async def typed_middleware(
    prompt: str | Sequence[Any],
    deps: MyDeps | None,
    *,
    ctx: ScopedContext | None = None
) -> str | Sequence[Any]:
    if deps:
        return f"[{deps.user}] {prompt}"
    return prompt
```

## Blocking

```python
from pydantic_ai_middleware import before_run, InputBlocked

@before_run
async def block_bad_words(prompt, deps, ctx):
    if "bad" in prompt.lower():
        raise InputBlocked("Bad word detected")
    return prompt
```

## When to Use Classes vs Decorators

**Use decorators when:**

- You have simple, single-purpose middleware
- You don't need to share state between hooks
- You want quick, inline definitions

**Use classes when:**

- You need multiple hooks in one middleware
- You need to share state between hooks
- You need configuration options
- You need to reuse middleware logic

# Permission Decisions

Structured ALLOW/DENY/ASK protocol for tool authorization, replacing the binary ToolBlocked approach.

## The Problem

Without permission decisions, `before_tool_call` can only:

1. Return modified args (allow)
2. Raise `ToolBlocked` (deny)

There's no way to defer a decision or provide structured reasons.

## The Solution

Return a `ToolPermissionResult` from `before_tool_call`:

```python
from pydantic_ai_middleware import (
    AgentMiddleware,
    ToolDecision,
    ToolPermissionResult,
)

class FileAccessControl(AgentMiddleware[None]):
    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        if tool_name == "delete_file":
            return ToolPermissionResult(
                decision=ToolDecision.ASK,
                reason=f"Delete {tool_args['path']}?",
            )
        if tool_name == "read_file":
            return ToolPermissionResult(
                decision=ToolDecision.ALLOW,
                modified_args={**tool_args, "audit": True},
            )
        return tool_args  # plain dict still works
```

## ToolDecision Enum

| Decision | Behavior |
|----------|----------|
| `ALLOW` | Tool call proceeds. Use `modified_args` if set. |
| `DENY` | Tool call is blocked. Raises `ToolBlocked` with `reason`. |
| `ASK` | Defers to `permission_handler` callback. If no handler, raises `ToolBlocked`. |

## ToolPermissionResult

```python
@dataclass
class ToolPermissionResult:
    decision: ToolDecision
    reason: str = ""
    modified_args: dict[str, Any] | None = None
```

- `decision` -- the authorization decision
- `reason` -- human-readable explanation (used in `ToolBlocked` message for DENY, passed to handler for ASK)
- `modified_args` -- optional replacement args (used with ALLOW and ASK when approved)

## Permission Handler

For `ASK` decisions, configure a `permission_handler` on the agent:

```python
async def approval_callback(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str,
) -> bool:
    # Your approval logic (UI prompt, admin check, etc.)
    return tool_name != "delete_file"

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[FileAccessControl()],
    permission_handler=approval_callback,
)
```

If the handler returns `True`, the tool call proceeds. If `False`, `ToolBlocked` is raised.

## Backwards Compatibility

Returning a plain `dict` from `before_tool_call` still works exactly as before:

```python
async def before_tool_call(self, tool_name, tool_args, deps, ctx):
    return tool_args  # same as ToolPermissionResult(ALLOW)
```

Raising `ToolBlocked` directly also still works:

```python
async def before_tool_call(self, tool_name, tool_args, deps, ctx):
    raise ToolBlocked(tool_name, "Not allowed")  # same as ToolPermissionResult(DENY)
```

## In Composite Middleware

Permission results flow through composite middleware:

- **MiddlewareChain** -- if any middleware returns `ToolPermissionResult`, it short-circuits the chain
- **ConditionalMiddleware** -- routes to selected branch, permission result passes through
- **MiddlewareToolset** -- processes the result (ALLOW/DENY/ASK logic)

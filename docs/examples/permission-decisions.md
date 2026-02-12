# Permission Decisions Example

Use structured ALLOW/DENY/ASK decisions for tool authorization.

## File Access Control

```python
from pydantic_ai_middleware import (
    AgentMiddleware,
    ToolDecision,
    ToolPermissionResult,
)

class FileAccessControl(AgentMiddleware[None]):
    """Control file operations with structured permissions."""

    SAFE_DIRS = {"/tmp", "/workspace"}
    DANGEROUS_TOOLS = {"delete_file", "write_file"}

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        path = tool_args.get("path", "")

        # Always allow reads
        if tool_name == "read_file":
            return ToolPermissionResult(decision=ToolDecision.ALLOW)

        # Block writes outside safe directories
        if not any(path.startswith(d) for d in self.SAFE_DIRS):
            return ToolPermissionResult(
                decision=ToolDecision.DENY,
                reason=f"Cannot write to {path} - not in safe directories",
            )

        # Ask for confirmation on destructive operations
        if tool_name in self.DANGEROUS_TOOLS:
            return ToolPermissionResult(
                decision=ToolDecision.ASK,
                reason=f"{tool_name}: {path}",
            )

        return tool_args
```

## Permission Handler

```python
async def cli_approval(tool_name, tool_args, reason):
    """Simple CLI-based approval."""
    print(f"\n--- Permission Required ---")
    print(f"Tool: {tool_name}")
    print(f"Reason: {reason}")
    response = input("Allow? (y/n): ")
    return response.lower() == "y"

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[FileAccessControl()],
    permission_handler=cli_approval,
)
```

## Role-Based Permissions

```python
from dataclasses import dataclass

@dataclass
class UserDeps:
    user_id: str
    roles: set[str]

class RoleBasedAccess(AgentMiddleware[UserDeps]):
    TOOL_ROLES = {
        "send_email": {"admin", "support"},
        "delete_file": {"admin"},
        "execute_code": {"admin", "developer"},
    }

    async def before_tool_call(self, tool_name, tool_args, deps, ctx):
        if deps is None:
            return ToolPermissionResult(
                decision=ToolDecision.DENY,
                reason="No user context",
            )

        required = self.TOOL_ROLES.get(tool_name)
        if required is None:
            return tool_args  # no restrictions

        if deps.roles & required:
            return ToolPermissionResult(decision=ToolDecision.ALLOW)

        return ToolPermissionResult(
            decision=ToolDecision.DENY,
            reason=f"User {deps.user_id} lacks roles: {required}",
        )
```

Source: [`examples/permission_decisions.py`](https://github.com/vstorm-co/pydantic-ai-middleware/blob/main/examples/permission_decisions.py)

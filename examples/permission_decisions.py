"""Example: Permission decisions - structured ALLOW/DENY/ASK for tool authorization.

This example demonstrates the ToolDecision/ToolPermissionResult protocol
for fine-grained tool access control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai_middleware import (
    AgentMiddleware,
    ScopedContext,
    ToolDecision,
    ToolPermissionResult,
)

# --- File access control ---


class FileAccessControl(AgentMiddleware[None]):
    """Control file operations with structured permissions."""

    SAFE_DIRS = {"/tmp", "/workspace"}
    DANGEROUS_TOOLS = {"delete_file", "write_file"}

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any] | ToolPermissionResult:
        path = tool_args.get("path", "")

        # Always allow reads
        if tool_name == "read_file":
            return ToolPermissionResult(decision=ToolDecision.ALLOW)

        # Block writes outside safe directories
        if not any(path.startswith(d) for d in self.SAFE_DIRS):
            return ToolPermissionResult(
                decision=ToolDecision.DENY,
                reason=f"Cannot write to {path} - outside safe directories",
            )

        # Ask for confirmation on destructive operations
        if tool_name in self.DANGEROUS_TOOLS:
            return ToolPermissionResult(
                decision=ToolDecision.ASK,
                reason=f"{tool_name}: {path}",
            )

        return tool_args


# --- Role-based access control ---


@dataclass
class UserDeps:
    user_id: str
    roles: set[str]


class RoleBasedAccess(AgentMiddleware[UserDeps]):
    """Enforce role-based access control for tools."""

    TOOL_ROLES: dict[str, set[str]] = {
        "send_email": {"admin", "support"},
        "delete_file": {"admin"},
        "execute_code": {"admin", "developer"},
    }

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: UserDeps | None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any] | ToolPermissionResult:
        if deps is None:
            return ToolPermissionResult(
                decision=ToolDecision.DENY,
                reason="No user context available",
            )

        required = self.TOOL_ROLES.get(tool_name)
        if required is None:
            return tool_args  # no restrictions for unlisted tools

        if deps.roles & required:
            return ToolPermissionResult(decision=ToolDecision.ALLOW)

        return ToolPermissionResult(
            decision=ToolDecision.DENY,
            reason=f"User {deps.user_id} needs one of: {required}",
        )


# --- Permission handler callback ---


async def cli_approval(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str,
) -> bool:
    """Simple approval callback (for use with ASK decisions)."""
    print("\n--- Permission Required ---")
    print(f"Tool: {tool_name}")
    print(f"Args: {tool_args}")
    print(f"Reason: {reason}")
    # In production, this could be a UI prompt, Slack message, etc.
    return False  # deny by default in this example


# --- Usage ---

print("Permission decision examples:")
print(f"  ToolDecision values: {[d.value for d in ToolDecision]}")
print(f"  FileAccessControl dangerous tools: {FileAccessControl.DANGEROUS_TOOLS}")
print(f"  RoleBasedAccess tool roles: {RoleBasedAccess.TOOL_ROLES}")

# To use with an agent:
#
# agent = MiddlewareAgent(
#     agent=base_agent,
#     middleware=[FileAccessControl()],
#     permission_handler=cli_approval,
# )

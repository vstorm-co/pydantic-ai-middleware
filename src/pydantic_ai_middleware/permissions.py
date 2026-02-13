"""Permission decision protocol for tool calls."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ToolDecision(Enum):
    """Decision for a tool call permission check.

    Attributes:
        ALLOW: Allow the tool call to proceed.
        DENY: Block the tool call.
        ASK: Defer the decision to a permission handler callback.
    """

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class ToolPermissionResult:
    """Structured result from a before_tool_call permission check.

    Instead of returning modified args (dict) or raising ToolBlocked,
    middleware can return a ToolPermissionResult with a structured decision.

    Examples:
        ```python
        # Allow with modified args
        return ToolPermissionResult(
            decision=ToolDecision.ALLOW,
            modified_args={"sanitized": True, **tool_args},
        )

        # Deny with reason
        return ToolPermissionResult(
            decision=ToolDecision.DENY,
            reason="Tool not authorized for this user",
        )

        # Ask the user/system
        return ToolPermissionResult(
            decision=ToolDecision.ASK,
            reason="This tool requires explicit approval",
        )
        ```
    """

    decision: ToolDecision
    reason: str = ""
    modified_args: dict[str, Any] | None = field(default=None)


@runtime_checkable
class PermissionHandler(Protocol):
    """Callback protocol for handling ASK decisions.

    Implement this protocol to provide a callback that decides whether
    to allow or deny a tool call when a middleware returns
    ``ToolPermissionResult(decision=ToolDecision.ASK)``.

    Examples:
        ```python
        async def cli_approval(
            tool_name: str,
            tool_args: dict[str, Any],
            reason: str,
        ) -> bool:
            print(f"Allow {tool_name}? Reason: {reason}")
            return input("(y/n): ").lower() == "y"
        ```
    """

    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        reason: str,
    ) -> Awaitable[bool]:
        """Decide whether to allow a tool call.

        Args:
            tool_name: The name of the tool requesting permission.
            tool_args: The arguments to the tool.
            reason: The reason the middleware is asking for permission.

        Returns:
            True to allow the tool call, False to deny it.
        """
        ...

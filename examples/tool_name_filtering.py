"""Example: Tool name filtering - scope middleware to specific tools.

This example demonstrates how to use tool_names to restrict which
tools trigger a middleware's before_tool_call / after_tool_call hooks.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareChain,
    ScopedContext,
    ToolBlocked,
    before_tool_call,
)

# --- Class-based approach ---


class EmailValidator(AgentMiddleware[None]):
    """Validates email-related tool calls only."""

    tool_names = {"send_email", "draft_email"}

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        if not tool_args.get("to"):
            raise ToolBlocked(tool_name, "Recipient email is required")
        if not tool_args.get("subject"):
            raise ToolBlocked(tool_name, "Subject is required")
        return tool_args


class LoggingMiddleware(AgentMiddleware[None]):
    """Logs all tool calls (no tool_names filter = matches everything)."""

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        print(f"[LOG] {tool_name}({tool_args})")
        return tool_args


# --- Decorator-based approach ---


@before_tool_call(tools={"execute_code"})
async def sandbox_check(
    tool_name: str,
    tool_args: dict[str, Any],
    deps: None,
    ctx: ScopedContext | None = None,
) -> dict[str, Any]:
    """Only fires for execute_code calls."""
    code = tool_args.get("code", "")
    if "rm -rf" in code or "os.system" in code:
        raise ToolBlocked(tool_name, "Dangerous code pattern detected")
    return tool_args


# --- Composing in a chain ---


chain = MiddlewareChain(
    [
        EmailValidator(),  # only send_email, draft_email
        sandbox_check,  # only execute_code
        LoggingMiddleware(),  # all tools
    ],
    name="tool_guards",
)

print(f"Pipeline: {chain}")
# Pipeline: tool_guards: EmailValidator -> _FunctionMiddleware -> LoggingMiddleware

# Verify filtering
print(f"EmailValidator handles 'send_email': {EmailValidator().tool_names}")
print(f"sandbox_check handles 'execute_code': {sandbox_check.tool_names}")
print(f"LoggingMiddleware handles all: {LoggingMiddleware().tool_names}")

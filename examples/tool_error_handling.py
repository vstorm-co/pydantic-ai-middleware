"""Example: on_tool_error hook - handle tool failures with context.

This example demonstrates how to use on_tool_error for error recovery,
logging, and exception transformation.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai_middleware import AgentMiddleware, ScopedContext, on_tool_error


# --- Class-based approach ---


class APIErrorHandler(AgentMiddleware[None]):
    """Convert API tool errors into user-friendly messages."""

    tool_names = {"web_search", "api_call"}

    async def on_tool_error(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        if isinstance(error, TimeoutError):
            query = tool_args.get("query", "unknown")
            return ConnectionError(f"{tool_name} timed out for query: {query}")

        if isinstance(error, PermissionError):
            return RuntimeError(f"Access denied for {tool_name}")

        # Re-raise original for unhandled error types
        return None


# --- Decorator-based approach ---


@on_tool_error(tools={"send_email"})
async def email_error_handler(
    tool_name: str,
    tool_args: dict[str, Any],
    error: Exception,
    deps: None,
    ctx: ScopedContext | None = None,
) -> Exception | None:
    """Convert email errors into a clear message."""
    recipient = tool_args.get("to", "unknown")
    return RuntimeError(f"Failed to email {recipient}: {error}")


@on_tool_error
async def global_error_logger(
    tool_name: str,
    tool_args: dict[str, Any],
    error: Exception,
    deps: None,
    ctx: ScopedContext | None = None,
) -> Exception | None:
    """Log all tool errors without modifying them."""
    print(f"[TOOL ERROR] {tool_name}({tool_args}): {type(error).__name__}: {error}")
    return None  # always re-raise the original


# --- Usage info ---

print("on_tool_error hook examples:")
print(f"  APIErrorHandler handles: {APIErrorHandler().tool_names}")
print(f"  email_error_handler handles: {email_error_handler.tool_names}")
print(f"  global_error_logger handles: {global_error_logger.tool_names}")

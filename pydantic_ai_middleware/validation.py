from collections.abc import Sequence
from typing import Any

from .base import AgentMiddleware


def validate_middleware_item(item: Any, context: str = "") -> AgentMiddleware[Any]:
    """Validate that an item is a middleware instance."""
    if not isinstance(item, AgentMiddleware):
        msg = f"Expected AgentMiddleware{context}, got {type(item).__name__}"
        raise TypeError(msg)
    return item


def validate_middleware_sequence(items: Sequence[Any]) -> list[AgentMiddleware[Any]]:
    """Validate and flatten a sequence of middleware."""
    result: list[AgentMiddleware[Any]] = []
    for item in items:
        if not isinstance(item, AgentMiddleware):
            raise TypeError(
                f"Expected AgentMiddleware instances in sequence, got {type(item).__name__}"
            )
        result.append(item)
    return result

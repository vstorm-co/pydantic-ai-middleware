"""Timeout utility for middleware hook calls."""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from .exceptions import MiddlewareTimeout

T = TypeVar("T")


async def call_with_timeout(
    coro: Any,
    timeout: float | None,
    middleware_name: str = "",
    hook_name: str = "",
) -> Any:
    """Call a coroutine with an optional timeout.

    Args:
        coro: The coroutine to await.
        timeout: Timeout in seconds, or None for no timeout.
        middleware_name: Name of the middleware (for error messages).
        hook_name: Name of the hook (for error messages).

    Returns:
        The result of the coroutine.

    Raises:
        MiddlewareTimeout: If the coroutine exceeds the timeout.
    """
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as e:
        raise MiddlewareTimeout(middleware_name, timeout, hook_name) from e

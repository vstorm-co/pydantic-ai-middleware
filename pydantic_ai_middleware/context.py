"""Middleware context for sharing data across middleware execution chain.

This module provides a context system that allows middleware to share data
while enforcing strict access controls:
- Each hook can only write to its own namespace
- Each hook can only read from previous hooks in the execution chain
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import Any


class HookType(IntEnum):
    """Execution order of middleware hooks.

    The integer values represent the execution order, used for access control.
    A hook can only read data from hooks with lower values (earlier in chain).
    """

    BEFORE_RUN = 1
    BEFORE_MODEL_REQUEST = 2
    BEFORE_TOOL_CALL = 3
    AFTER_TOOL_CALL = 4
    AFTER_RUN = 5
    ON_ERROR = 6  # Special: can read all


class ContextAccessError(Exception):
    """Raised when middleware attempts unauthorized context access."""

    pass


class ScopedContext:
    """A scoped view of MiddlewareContext for a specific hook.

    Enforces access control:
    - Can only write to current hook's namespace
    - Can only read from current and previous hooks' namespaces
    """

    def __init__(self, parent: MiddlewareContext, current_hook: HookType) -> None:
        """Initialize scoped context.

        Args:
            parent: The parent MiddlewareContext.
            current_hook: The hook type this scope is for.
        """
        self._parent = parent
        self._current_hook = current_hook

    @property
    def config(self) -> Mapping[str, Any]:
        """Read-only access to global configuration."""
        return self._parent.config

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only access to execution metadata."""
        return self._parent.metadata

    @property
    def current_hook(self) -> HookType:
        """The current hook type for this scoped context."""
        return self._current_hook

    def _can_read(self, hook: HookType) -> bool:
        """Check if current hook can read from the specified hook."""
        # ON_ERROR can read everything
        if self._current_hook == HookType.ON_ERROR:
            return True
        # Can read from hooks earlier or equal in the chain
        return hook <= self._current_hook

    def set(self, key: str, value: Any) -> None:
        """Set a value in the current hook's namespace.

        Args:
            key: The key to set.
            value: The value to store.
        """
        self._parent._set_hook_data(self._current_hook, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the current hook's namespace.

        Args:
            key: The key to retrieve.
            default: Default value if key not found.

        Returns:
            The stored value or default.
        """
        return self.get_from(self._current_hook, key, default)

    def get_from(self, hook: HookType, key: str, default: Any = None) -> Any:
        """Get a value from a specific hook's namespace.

        Args:
            hook: The hook namespace to read from.
            key: The key to retrieve.
            default: Default value if key not found.

        Returns:
            The stored value or default.

        Raises:
            ContextAccessError: If current hook cannot read from specified hook.
        """
        if not self._can_read(hook):
            raise ContextAccessError(
                f"Hook '{self._current_hook.name}' cannot read from "
                f"'{hook.name}' (later in execution chain)"
            )
        return self._parent._get_hook_data(hook, key, default)

    def get_all_from(self, hook: HookType) -> Mapping[str, Any]:
        """Get all data from a specific hook's namespace.

        Args:
            hook: The hook namespace to read from.

        Returns:
            Read-only mapping of all data in the namespace.

        Raises:
            ContextAccessError: If current hook cannot read from specified hook.
        """
        if not self._can_read(hook):
            raise ContextAccessError(
                f"Hook '{self._current_hook.name}' cannot read from "
                f"'{hook.name}' (later in execution chain)"
            )
        return self._parent._get_all_hook_data(hook)

    def has_key(self, key: str) -> bool:
        """Check if key exists in current hook's namespace.

        Args:
            key: The key to check.

        Returns:
            True if key exists.
        """
        return self.has_key_in(self._current_hook, key)

    def has_key_in(self, hook: HookType, key: str) -> bool:
        """Check if key exists in a specific hook's namespace.

        Args:
            hook: The hook namespace to check.
            key: The key to check.

        Returns:
            True if key exists.

        Raises:
            ContextAccessError: If current hook cannot read from specified hook.
        """
        if not self._can_read(hook):
            raise ContextAccessError(
                f"Hook '{self._current_hook.name}' cannot read from "
                f"'{hook.name}' (later in execution chain)"
            )
        return self._parent._has_hook_key(hook, key)


class MiddlewareContext:
    """Context object for sharing data across middleware execution chain.

    This context provides:
    - Immutable global configuration
    - Execution metadata (timestamps, prompts, etc.)
    - Namespaced storage per hook type with access control

    Access control is enforced through ScopedContext views:
    - Each hook gets a scoped view via `for_hook()`
    - Hooks can only write to their own namespace
    - Hooks can only read from previous hooks in the chain

    Example:
        ```python
        # Create context with config
        ctx = MiddlewareContext(config={'rate_limit': 100})

        # Get scoped view for before_run hook
        scoped = ctx.for_hook(HookType.BEFORE_RUN)
        scoped.set('user_intent', 'question')  # OK

        # Get scoped view for after_run hook
        after_scoped = ctx.for_hook(HookType.AFTER_RUN)
        intent = after_scoped.get_from(HookType.BEFORE_RUN, 'user_intent')  # OK

        # This would raise ContextAccessError:
        # scoped.get_from(HookType.AFTER_RUN, 'some_key')  # Error!
        ```
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize middleware context.

        Args:
            config: Global configuration (becomes immutable).
            metadata: Initial execution metadata.
        """
        self._config: dict[str, Any] = dict(config) if config else {}
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}

        # Namespaced storage for each hook type
        self._hook_data: dict[HookType, dict[str, Any]] = {hook: {} for hook in HookType}

    @property
    def config(self) -> Mapping[str, Any]:
        """Read-only access to global configuration."""
        return self._config

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only access to execution metadata."""
        return self._metadata

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value (internal use by MiddlewareAgent).

        Args:
            key: The metadata key.
            value: The value to store.
        """
        self._metadata[key] = value

    def for_hook(self, hook: HookType) -> ScopedContext:
        """Get a scoped context view for a specific hook.

        Args:
            hook: The hook type to create scope for.

        Returns:
            A ScopedContext with appropriate access controls.
        """
        return ScopedContext(self, hook)

    def _set_hook_data(self, hook: HookType, key: str, value: Any) -> None:
        """Internal: Set data in a hook's namespace."""
        self._hook_data[hook][key] = value

    def _get_hook_data(self, hook: HookType, key: str, default: Any = None) -> Any:
        """Internal: Get data from a hook's namespace."""
        return self._hook_data[hook].get(key, default)

    def _get_all_hook_data(self, hook: HookType) -> Mapping[str, Any]:
        """Internal: Get all data from a hook's namespace."""
        return self._hook_data[hook]

    def _has_hook_key(self, hook: HookType, key: str) -> bool:
        """Internal: Check if key exists in hook's namespace."""
        return key in self._hook_data[hook]

    def clone(self) -> MiddlewareContext:
        """Create a shallow clone of this context.

        Used for parallel middleware execution where each parallel
        middleware gets a clone to prevent race conditions.

        Returns:
            A new MiddlewareContext with copied data.
        """
        new_ctx = MiddlewareContext(
            config=dict(self._config),
            metadata=dict(self._metadata),
        )
        # Deep copy hook data
        for hook, data in self._hook_data.items():
            new_ctx._hook_data[hook] = dict(data)
        return new_ctx

    def merge_from(self, other: MiddlewareContext, hook: HookType) -> None:
        """Merge data from another context's hook namespace.

        Used after parallel execution to merge results back.

        Args:
            other: The context to merge from.
            hook: The hook namespace to merge.
        """
        self._hook_data[hook].update(other._hook_data[hook])

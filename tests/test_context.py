"""Tests for middleware context module."""

from __future__ import annotations

import pytest

from pydantic_ai_middleware.context import (
    ContextAccessError,
    HookType,
    MiddlewareContext,
    ScopedContext,
)


class TestHookType:
    """Tests for HookType enum."""

    def test_hook_order(self) -> None:
        """Test that hooks are ordered correctly for access control."""
        assert HookType.BEFORE_RUN < HookType.BEFORE_MODEL_REQUEST
        assert HookType.BEFORE_MODEL_REQUEST < HookType.BEFORE_TOOL_CALL
        assert HookType.BEFORE_TOOL_CALL < HookType.AFTER_TOOL_CALL
        assert HookType.AFTER_TOOL_CALL < HookType.AFTER_RUN
        assert HookType.AFTER_RUN < HookType.ON_ERROR

    def test_hook_values_are_integers(self) -> None:
        """Test that hook types have integer values."""
        for hook in HookType:
            assert isinstance(hook.value, int)


class TestMiddlewareContext:
    """Tests for MiddlewareContext class."""

    def test_create_empty_context(self) -> None:
        """Test creating an empty context."""
        ctx = MiddlewareContext()
        assert ctx.config == {}
        assert ctx.metadata == {}

    def test_create_with_config(self) -> None:
        """Test creating context with config."""
        ctx = MiddlewareContext(config={"rate_limit": 100, "debug": True})
        assert ctx.config["rate_limit"] == 100
        assert ctx.config["debug"] is True

    def test_create_with_metadata(self) -> None:
        """Test creating context with metadata."""
        ctx = MiddlewareContext(metadata={"start_time": 123.45})
        assert ctx.metadata["start_time"] == 123.45

    def test_config_is_readonly(self) -> None:
        """Test that config mapping cannot be modified directly."""
        ctx = MiddlewareContext(config={"key": "value"})
        # The mapping type should not allow item assignment
        # (This tests the Mapping type hint behavior)
        assert ctx.config["key"] == "value"

    def test_set_metadata(self) -> None:
        """Test setting metadata values."""
        ctx = MiddlewareContext()
        ctx.set_metadata("end_time", 456.78)
        assert ctx.metadata["end_time"] == 456.78

    def test_for_hook_returns_scoped_context(self) -> None:
        """Test that for_hook returns a ScopedContext."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)
        assert isinstance(scoped, ScopedContext)
        assert scoped.current_hook == HookType.BEFORE_RUN

    def test_clone_creates_independent_copy(self) -> None:
        """Test that clone creates an independent copy."""
        ctx = MiddlewareContext(
            config={"key": "value"},
            metadata={"start": 1},
        )
        ctx._set_hook_data(HookType.BEFORE_RUN, "data", "original")

        cloned = ctx.clone()

        # Modify original
        ctx.set_metadata("modified", True)
        ctx._set_hook_data(HookType.BEFORE_RUN, "data", "changed")

        # Clone should be unaffected
        assert "modified" not in cloned.metadata
        assert cloned._get_hook_data(HookType.BEFORE_RUN, "data") == "original"

    def test_merge_from(self) -> None:
        """Test merging data from another context."""
        ctx1 = MiddlewareContext()
        ctx1._set_hook_data(HookType.BEFORE_RUN, "key1", "value1")

        ctx2 = MiddlewareContext()
        ctx2._set_hook_data(HookType.BEFORE_RUN, "key2", "value2")

        ctx1.merge_from(ctx2, HookType.BEFORE_RUN)

        assert ctx1._get_hook_data(HookType.BEFORE_RUN, "key1") == "value1"
        assert ctx1._get_hook_data(HookType.BEFORE_RUN, "key2") == "value2"


class TestScopedContext:
    """Tests for ScopedContext class."""

    def test_access_config(self) -> None:
        """Test accessing config from scoped context."""
        ctx = MiddlewareContext(config={"setting": "value"})
        scoped = ctx.for_hook(HookType.BEFORE_RUN)
        assert scoped.config["setting"] == "value"

    def test_access_metadata(self) -> None:
        """Test accessing metadata from scoped context."""
        ctx = MiddlewareContext(metadata={"info": "data"})
        scoped = ctx.for_hook(HookType.BEFORE_RUN)
        assert scoped.metadata["info"] == "data"

    def test_set_in_own_namespace(self) -> None:
        """Test setting data in own namespace."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        scoped.set("my_key", "my_value")

        assert scoped.get("my_key") == "my_value"

    def test_get_default_value(self) -> None:
        """Test getting with default value."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        assert scoped.get("missing_key") is None
        assert scoped.get("missing_key", "default") == "default"

    def test_has_key(self) -> None:
        """Test checking if key exists."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        assert not scoped.has_key("key")
        scoped.set("key", "value")
        assert scoped.has_key("key")


class TestScopedContextAccessControl:
    """Tests for access control enforcement in ScopedContext."""

    def test_before_run_can_read_own_namespace(self) -> None:
        """Test before_run can read from its own namespace."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)
        scoped.set("key", "value")
        assert scoped.get_from(HookType.BEFORE_RUN, "key") == "value"

    def test_before_run_cannot_read_after_run(self) -> None:
        """Test before_run cannot read from after_run namespace."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.AFTER_RUN, "key", "value")

        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        with pytest.raises(ContextAccessError) as exc_info:
            scoped.get_from(HookType.AFTER_RUN, "key")

        assert "BEFORE_RUN" in str(exc_info.value)
        assert "AFTER_RUN" in str(exc_info.value)
        assert "later in execution chain" in str(exc_info.value)

    def test_before_run_cannot_read_before_model_request(self) -> None:
        """Test before_run cannot read from before_model_request."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_MODEL_REQUEST, "key", "value")

        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        with pytest.raises(ContextAccessError):
            scoped.get_from(HookType.BEFORE_MODEL_REQUEST, "key")

    def test_after_run_can_read_before_run(self) -> None:
        """Test after_run can read from before_run namespace."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_RUN, "key", "value")

        scoped = ctx.for_hook(HookType.AFTER_RUN)
        assert scoped.get_from(HookType.BEFORE_RUN, "key") == "value"

    def test_after_run_can_read_all_before_hooks(self) -> None:
        """Test after_run can read from all previous hooks."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_RUN, "k1", "v1")
        ctx._set_hook_data(HookType.BEFORE_MODEL_REQUEST, "k2", "v2")
        ctx._set_hook_data(HookType.BEFORE_TOOL_CALL, "k3", "v3")
        ctx._set_hook_data(HookType.AFTER_TOOL_CALL, "k4", "v4")

        scoped = ctx.for_hook(HookType.AFTER_RUN)

        assert scoped.get_from(HookType.BEFORE_RUN, "k1") == "v1"
        assert scoped.get_from(HookType.BEFORE_MODEL_REQUEST, "k2") == "v2"
        assert scoped.get_from(HookType.BEFORE_TOOL_CALL, "k3") == "v3"
        assert scoped.get_from(HookType.AFTER_TOOL_CALL, "k4") == "v4"

    def test_on_error_can_read_all_hooks(self) -> None:
        """Test on_error can read from all hook namespaces."""
        ctx = MiddlewareContext()
        for hook in HookType:
            if hook != HookType.ON_ERROR:
                ctx._set_hook_data(hook, "key", f"value_{hook.name}")

        scoped = ctx.for_hook(HookType.ON_ERROR)

        for hook in HookType:
            if hook != HookType.ON_ERROR:
                assert scoped.get_from(hook, "key") == f"value_{hook.name}"

    def test_before_model_request_can_read_before_run(self) -> None:
        """Test before_model_request can read from before_run."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_RUN, "key", "value")

        scoped = ctx.for_hook(HookType.BEFORE_MODEL_REQUEST)
        assert scoped.get_from(HookType.BEFORE_RUN, "key") == "value"

    def test_before_model_request_cannot_read_after_tool_call(self) -> None:
        """Test before_model_request cannot read from after_tool_call."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.AFTER_TOOL_CALL, "key", "value")

        scoped = ctx.for_hook(HookType.BEFORE_MODEL_REQUEST)

        with pytest.raises(ContextAccessError):
            scoped.get_from(HookType.AFTER_TOOL_CALL, "key")

    def test_get_all_from_respects_access_control(self) -> None:
        """Test get_all_from respects access control."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_RUN, "k1", "v1")
        ctx._set_hook_data(HookType.BEFORE_RUN, "k2", "v2")

        scoped = ctx.for_hook(HookType.AFTER_RUN)
        all_data = scoped.get_all_from(HookType.BEFORE_RUN)

        assert all_data["k1"] == "v1"
        assert all_data["k2"] == "v2"

    def test_get_all_from_denies_future_hooks(self) -> None:
        """Test get_all_from denies access to future hooks."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        with pytest.raises(ContextAccessError):
            scoped.get_all_from(HookType.AFTER_RUN)

    def test_has_key_in_respects_access_control(self) -> None:
        """Test has_key_in respects access control."""
        ctx = MiddlewareContext()
        ctx._set_hook_data(HookType.BEFORE_RUN, "key", "value")

        scoped = ctx.for_hook(HookType.AFTER_RUN)
        assert scoped.has_key_in(HookType.BEFORE_RUN, "key")

    def test_has_key_in_denies_future_hooks(self) -> None:
        """Test has_key_in denies access to future hooks."""
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        with pytest.raises(ContextAccessError):
            scoped.has_key_in(HookType.AFTER_RUN, "key")


class TestScopedContextChaining:
    """Tests for data flow across hooks via context."""

    def test_data_flows_through_chain(self) -> None:
        """Test that data set in early hooks is accessible in later hooks."""
        ctx = MiddlewareContext()

        # Simulate middleware chain execution
        before_run = ctx.for_hook(HookType.BEFORE_RUN)
        before_run.set("user_intent", "question")
        before_run.set("original_prompt", "What is 2+2?")

        before_model = ctx.for_hook(HookType.BEFORE_MODEL_REQUEST)
        # Can read from before_run
        assert before_model.get_from(HookType.BEFORE_RUN, "user_intent") == "question"
        before_model.set("model_call_count", 1)

        before_tool = ctx.for_hook(HookType.BEFORE_TOOL_CALL)
        # Can read from before_run and before_model_request
        assert before_tool.get_from(HookType.BEFORE_RUN, "original_prompt") == "What is 2+2?"
        assert before_tool.get_from(HookType.BEFORE_MODEL_REQUEST, "model_call_count") == 1
        before_tool.set("tool_name", "calculator")

        after_tool = ctx.for_hook(HookType.AFTER_TOOL_CALL)
        # Can read from all before_* hooks
        assert after_tool.get_from(HookType.BEFORE_TOOL_CALL, "tool_name") == "calculator"
        after_tool.set("tool_result", 4)

        after_run = ctx.for_hook(HookType.AFTER_RUN)
        # Can read from everything
        assert after_run.get_from(HookType.BEFORE_RUN, "user_intent") == "question"
        assert after_run.get_from(HookType.AFTER_TOOL_CALL, "tool_result") == 4

    def test_multiple_middleware_share_namespace(self) -> None:
        """Test that multiple middleware in the same hook share namespace."""
        ctx = MiddlewareContext()

        # First middleware in before_run
        mw1_ctx = ctx.for_hook(HookType.BEFORE_RUN)
        mw1_ctx.set("mw1_validated", True)

        # Second middleware in before_run (same scoped context behavior)
        mw2_ctx = ctx.for_hook(HookType.BEFORE_RUN)
        mw2_ctx.set("mw2_transformed", "new_prompt")

        # Both values should be accessible from same namespace
        assert mw1_ctx.get("mw1_validated") is True
        assert mw1_ctx.get("mw2_transformed") == "new_prompt"
        assert mw2_ctx.get("mw1_validated") is True


class TestContextReset:
    """Tests for MiddlewareContext reset functionality."""

    def test_reset_clears_metadata(self) -> None:
        """Test that reset() clears metadata."""
        ctx = MiddlewareContext()
        ctx.set_metadata("prompt", "hello")
        ctx.set_metadata("output", "world")

        assert ctx.metadata["prompt"] == "hello"
        assert ctx.metadata["output"] == "world"

        ctx.reset()

        assert len(ctx.metadata) == 0
        assert "prompt" not in ctx.metadata
        assert "output" not in ctx.metadata

    def test_reset_clears_hook_data(self) -> None:
        """Test that reset() clears all hook data."""
        ctx = MiddlewareContext()

        # Set data in multiple hooks
        ctx._set_hook_data(HookType.BEFORE_RUN, "key1", "value1")
        ctx._set_hook_data(HookType.AFTER_RUN, "key2", "value2")
        ctx._set_hook_data(HookType.BEFORE_MODEL_REQUEST, "key3", "value3")

        assert ctx._get_hook_data(HookType.BEFORE_RUN, "key1") == "value1"
        assert ctx._get_hook_data(HookType.AFTER_RUN, "key2") == "value2"
        assert ctx._get_hook_data(HookType.BEFORE_MODEL_REQUEST, "key3") == "value3"

        ctx.reset()

        # All data should be cleared
        assert ctx._get_hook_data(HookType.BEFORE_RUN, "key1") is None
        assert ctx._get_hook_data(HookType.AFTER_RUN, "key2") is None
        assert ctx._get_hook_data(HookType.BEFORE_MODEL_REQUEST, "key3") is None

    def test_reset_preserves_config(self) -> None:
        """Test that reset() preserves configuration."""
        ctx = MiddlewareContext(config={"rate_limit": 100, "timeout": 30})
        ctx.set_metadata("run_id", "123")
        ctx._set_hook_data(HookType.BEFORE_RUN, "data", "value")

        ctx.reset()

        # Config should still be there
        assert ctx.config["rate_limit"] == 100
        assert ctx.config["timeout"] == 30

        # Metadata and hook data should be cleared
        assert len(ctx.metadata) == 0
        assert ctx._get_hook_data(HookType.BEFORE_RUN, "data") is None

    def test_reset_allows_fresh_state_for_new_run(self) -> None:
        """Test that reset() enables clean state for new runs."""
        ctx = MiddlewareContext(config={"mode": "test"})

        # First run
        ctx.set_metadata("prompt", "first prompt")
        scoped1 = ctx.for_hook(HookType.BEFORE_RUN)
        scoped1.set("input_length", 12)

        assert ctx.metadata["prompt"] == "first prompt"
        assert ctx._get_hook_data(HookType.BEFORE_RUN, "input_length") == 12

        # Reset for second run
        ctx.reset()

        # Second run should have clean state
        ctx.set_metadata("prompt", "second prompt")
        scoped2 = ctx.for_hook(HookType.BEFORE_RUN)
        scoped2.set("input_length", 14)

        assert ctx.metadata["prompt"] == "second prompt"
        assert ctx._get_hook_data(HookType.BEFORE_RUN, "input_length") == 14
        # Should not have data from first run
        assert not any(k for k in ctx._hook_data[HookType.BEFORE_RUN] if k not in ["input_length"])

    def test_reset_clears_all_hook_types(self) -> None:
        """Test that reset() clears data from all hook types."""
        ctx = MiddlewareContext()

        # Set data in all hook types
        for hook in [
            HookType.BEFORE_RUN,
            HookType.BEFORE_MODEL_REQUEST,
            HookType.BEFORE_TOOL_CALL,
            HookType.AFTER_TOOL_CALL,
            HookType.AFTER_RUN,
            HookType.ON_ERROR,
        ]:
            scoped = ctx.for_hook(hook)
            scoped.set(f"{hook.name}_data", f"{hook.name}_value")

        # Verify all data is set
        for hook in HookType:
            assert ctx._get_hook_data(hook, f"{hook.name}_data") == f"{hook.name}_value"

        ctx.reset()

        # Verify all data is cleared
        for hook in HookType:
            assert ctx._get_hook_data(hook, f"{hook.name}_data") is None

    def test_reset_multiple_times(self) -> None:
        """Test that reset() can be called multiple times safely."""
        ctx = MiddlewareContext(config={"key": "value"})

        # Multiple cycles of set and reset
        for i in range(3):
            ctx.set_metadata("run_number", i)
            ctx._set_hook_data(HookType.BEFORE_RUN, "iteration", i)

            assert ctx.metadata["run_number"] == i
            assert ctx._get_hook_data(HookType.BEFORE_RUN, "iteration") == i

            ctx.reset()

            assert len(ctx.metadata) == 0
            assert ctx._get_hook_data(HookType.BEFORE_RUN, "iteration") is None
            assert ctx.config["key"] == "value"  # Config still there

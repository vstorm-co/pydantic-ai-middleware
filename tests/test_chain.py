"""Tests for MiddlewareChain."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage

from pydantic_ai_middleware import (
    AgentMiddleware,
    HookType,
    MiddlewareChain,
    MiddlewareContext,
    ScopedContext,
)


class TrackingMiddleware(AgentMiddleware[None]):
    """Middleware that tracks method calls and their order."""

    call_order: list[str] = []  # Class-level to track order across instances

    def __init__(self, name: str) -> None:
        self.name = name
        self.before_run_called = False
        self.after_run_called = False
        self.before_model_called = False
        self.before_tool_called = False
        self.after_tool_called = False
        self.on_error_called = False

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        self.before_run_called = True
        TrackingMiddleware.call_order.append(f"{self.name}:before_run")
        return f"[{self.name}]{prompt}"

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        self.after_run_called = True
        TrackingMiddleware.call_order.append(f"{self.name}:after_run")
        return f"[{self.name}]{output}"

    async def before_model_request(
        self,
        messages: list[ModelMessage],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> list[ModelMessage]:
        self.before_model_called = True
        TrackingMiddleware.call_order.append(f"{self.name}:before_model")
        return messages

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        self.before_tool_called = True
        TrackingMiddleware.call_order.append(f"{self.name}:before_tool")
        return {**tool_args, self.name: True}

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        self.after_tool_called = True
        TrackingMiddleware.call_order.append(f"{self.name}:after_tool")
        return f"[{self.name}]{result}"

    async def on_error(
        self,
        error: Exception,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        self.on_error_called = True
        TrackingMiddleware.call_order.append(f"{self.name}:on_error")
        return None


class ErrorHandlerMiddleware(AgentMiddleware[None]):
    """Middleware that handles errors by converting them."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def on_error(
        self,
        error: Exception,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Exception | None:
        return ValueError(f"Handled by {self.name}: {error}")


@pytest.fixture(autouse=True)
def reset_call_order():
    """Reset the call order list before each test."""
    TrackingMiddleware.call_order = []
    yield


class TestMiddlewareChainInit:
    """Tests for MiddlewareChain initialization."""

    def test_init_empty(self) -> None:
        """Test creating an empty chain."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        assert len(chain) == 0
        assert chain.middleware == []

    def test_init_with_middleware(self) -> None:
        """Test creating a chain with middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])
        assert len(chain) == 2
        assert chain.middleware == [mw1, mw2]

    def test_init_with_name(self) -> None:
        """Test creating a chain with a custom name."""
        chain: MiddlewareChain[None] = MiddlewareChain(name="my_chain")
        assert chain.name == "my_chain"

    def test_init_default_name(self) -> None:
        """Test default name includes middleware count."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])
        assert "MiddlewareChain" in chain.name
        assert "2" in chain.name

    def test_init_flattens_nested_chains(self) -> None:
        """Test that nested chains are automatically flattened."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        inner_chain = MiddlewareChain([mw1, mw2])

        mw3 = TrackingMiddleware("mw3")
        outer_chain = MiddlewareChain([inner_chain, mw3])

        assert len(outer_chain) == 3
        assert outer_chain.middleware == [mw1, mw2, mw3]


class TestMiddlewareChainModification:
    """Tests for MiddlewareChain modification methods."""

    def test_add_middleware(self) -> None:
        """Test adding middleware to the chain."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        mw = TrackingMiddleware("mw1")
        result = chain.add(mw)

        assert len(chain) == 1
        assert mw in chain
        assert result is chain  # Returns self for chaining

    def test_add_chain_flattens(self) -> None:
        """Test adding a chain flattens it."""
        chain1: MiddlewareChain[None] = MiddlewareChain()
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain2 = MiddlewareChain([mw1, mw2])

        chain1.add(chain2)

        assert len(chain1) == 2
        assert mw1 in chain1
        assert mw2 in chain1

    def test_insert_middleware(self) -> None:
        """Test inserting middleware at a specific position."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        chain = MiddlewareChain([mw1, mw3])

        result = chain.insert(1, mw2)

        assert chain.middleware == [mw1, mw2, mw3]
        assert result is chain

    def test_insert_at_beginning(self) -> None:
        """Test inserting middleware at the beginning."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw2])

        chain.insert(0, mw1)

        assert chain.middleware == [mw1, mw2]

    def test_insert_chain_flattens(self) -> None:
        """Test inserting a chain flattens it in order."""
        mw1 = TrackingMiddleware("mw1")
        mw4 = TrackingMiddleware("mw4")
        chain1 = MiddlewareChain([mw1, mw4])

        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        chain2 = MiddlewareChain([mw2, mw3])

        chain1.insert(1, chain2)

        assert chain1.middleware == [mw1, mw2, mw3, mw4]

    def test_remove_middleware(self) -> None:
        """Test removing middleware from the chain."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        result = chain.remove(mw1)

        assert len(chain) == 1
        assert mw1 not in chain
        assert mw2 in chain
        assert result is chain

    def test_remove_nonexistent_raises(self) -> None:
        """Test removing nonexistent middleware raises ValueError."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        mw = TrackingMiddleware("mw1")

        with pytest.raises(ValueError):
            chain.remove(mw)

    def test_pop_default(self) -> None:
        """Test pop removes and returns last middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        popped = chain.pop()

        assert popped is mw2
        assert len(chain) == 1
        assert chain.middleware == [mw1]

    def test_pop_with_index(self) -> None:
        """Test pop with specific index."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        popped = chain.pop(0)

        assert popped is mw1
        assert len(chain) == 1
        assert chain.middleware == [mw2]

    def test_pop_empty_raises(self) -> None:
        """Test pop on empty chain raises IndexError."""
        chain: MiddlewareChain[None] = MiddlewareChain()

        with pytest.raises(IndexError):
            chain.pop()

    def test_replace_middleware(self) -> None:
        """Test replacing middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        chain = MiddlewareChain([mw1, mw2])

        result = chain.replace(mw2, mw3)

        assert chain.middleware == [mw1, mw3]
        assert result is chain

    def test_replace_nonexistent_raises(self) -> None:
        """Test replacing nonexistent middleware raises ValueError."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1])

        with pytest.raises(ValueError):
            chain.replace(mw2, mw1)

    def test_clear(self) -> None:
        """Test clearing all middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        result = chain.clear()

        assert len(chain) == 0
        assert chain.middleware == []
        assert result is chain

    def test_copy(self) -> None:
        """Test copying a chain creates independent copy."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2], name="original")

        copied = chain.copy()

        assert len(copied) == 2
        assert copied.middleware == [mw1, mw2]

        # Modifying copy doesn't affect original
        mw3 = TrackingMiddleware("mw3")
        copied.add(mw3)
        assert len(chain) == 2
        assert len(copied) == 3


class TestMiddlewareChainOperators:
    """Tests for MiddlewareChain operator overloads."""

    def test_add_operator(self) -> None:
        """Test + operator creates new chain."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain1 = MiddlewareChain([mw1])
        chain2 = MiddlewareChain([mw2])

        combined = chain1 + chain2

        assert len(combined) == 2
        assert combined.middleware == [mw1, mw2]
        # Original chains unchanged
        assert len(chain1) == 1
        assert len(chain2) == 1

    def test_add_operator_with_single_middleware(self) -> None:
        """Test + operator with single middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1])

        combined = chain + mw2

        assert len(combined) == 2
        assert combined.middleware == [mw1, mw2]

    def test_iadd_operator(self) -> None:
        """Test += operator modifies in place."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1])

        chain += mw2

        assert len(chain) == 2
        assert chain.middleware == [mw1, mw2]

    def test_iadd_operator_with_chain(self) -> None:
        """Test += operator with another chain."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain1 = MiddlewareChain([mw1])
        chain2 = MiddlewareChain([mw2])

        chain1 += chain2

        assert len(chain1) == 2
        assert chain1.middleware == [mw1, mw2]


class TestMiddlewareChainCollection:
    """Tests for MiddlewareChain collection interface."""

    def test_len(self) -> None:
        """Test len() returns correct count."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])
        assert len(chain) == 2

    def test_bool_empty(self) -> None:
        """Test bool() on empty chain."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        assert not chain

    def test_bool_nonempty(self) -> None:
        """Test bool() on non-empty chain."""
        mw = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw])
        assert chain

    def test_getitem_index(self) -> None:
        """Test [] indexing."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        assert chain[0] is mw1
        assert chain[1] is mw2
        assert chain[-1] is mw2

    def test_getitem_slice(self) -> None:
        """Test [] slicing."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        chain = MiddlewareChain([mw1, mw2, mw3])

        assert chain[1:] == [mw2, mw3]
        assert chain[:2] == [mw1, mw2]

    def test_iter(self) -> None:
        """Test iteration."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        items = list(chain)
        assert items == [mw1, mw2]

    def test_contains(self) -> None:
        """Test in operator."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1])

        assert mw1 in chain
        assert mw2 not in chain


class TestMiddlewareChainStringRepresentation:
    """Tests for MiddlewareChain string representation."""

    def test_repr_empty(self) -> None:
        """Test repr of empty chain."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        assert repr(chain) == "MiddlewareChain([])"

    def test_repr_with_middleware(self) -> None:
        """Test repr with middleware."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])
        r = repr(chain)

        assert "MiddlewareChain" in r
        assert "TrackingMiddleware" in r

    def test_str_empty(self) -> None:
        """Test str of empty chain."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        s = str(chain)

        assert "(empty)" in s

    def test_str_with_middleware(self) -> None:
        """Test str with middleware shows flow."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])
        s = str(chain)

        assert "TrackingMiddleware" in s
        assert "->" in s


class TestMiddlewareChainBeforeRun:
    """Tests for MiddlewareChain.before_run()."""

    async def test_before_run_empty_chain(self) -> None:
        """Test before_run with empty chain returns prompt unchanged."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        result = await chain.before_run("test", None)
        assert result == "test"

    async def test_before_run_single_middleware(self) -> None:
        """Test before_run with single middleware."""
        mw = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw])

        result = await chain.before_run("test", None)

        assert result == "[mw1]test"
        assert mw.before_run_called

    async def test_before_run_multiple_middleware_order(self) -> None:
        """Test before_run executes middleware in order."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        chain = MiddlewareChain([mw1, mw2, mw3])

        result = await chain.before_run("test", None)

        assert result == "[mw3][mw2][mw1]test"
        assert TrackingMiddleware.call_order == [
            "mw1:before_run",
            "mw2:before_run",
            "mw3:before_run",
        ]

    async def test_before_run_with_context(self) -> None:
        """Test before_run passes context to middleware."""
        mw = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw])
        ctx = MiddlewareContext()
        scoped = ctx.for_hook(HookType.BEFORE_RUN)

        result = await chain.before_run("test", None, ctx=scoped)

        assert result == "[mw1]test"


class TestMiddlewareChainAfterRun:
    """Tests for MiddlewareChain.after_run()."""

    async def test_after_run_empty_chain(self) -> None:
        """Test after_run with empty chain returns output unchanged."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        result = await chain.after_run("prompt", "output", None)
        assert result == "output"

    async def test_after_run_single_middleware(self) -> None:
        """Test after_run with single middleware."""
        mw = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw])

        result = await chain.after_run("prompt", "output", None)

        assert result == "[mw1]output"
        assert mw.after_run_called

    async def test_after_run_reverse_order(self) -> None:
        """Test after_run executes middleware in reverse order."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        chain = MiddlewareChain([mw1, mw2, mw3])

        result = await chain.after_run("prompt", "output", None)

        # Reverse order: mw3 -> mw2 -> mw1
        assert result == "[mw1][mw2][mw3]output"
        assert TrackingMiddleware.call_order == [
            "mw3:after_run",
            "mw2:after_run",
            "mw1:after_run",
        ]


class TestMiddlewareChainBeforeModelRequest:
    """Tests for MiddlewareChain.before_model_request()."""

    async def test_before_model_request_empty_chain(self) -> None:
        """Test before_model_request with empty chain."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        messages: list[ModelMessage] = []
        result = await chain.before_model_request(messages, None)
        assert result == []

    async def test_before_model_request_order(self) -> None:
        """Test before_model_request executes in order."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        messages: list[ModelMessage] = []
        await chain.before_model_request(messages, None)

        assert TrackingMiddleware.call_order == [
            "mw1:before_model",
            "mw2:before_model",
        ]


class TestMiddlewareChainToolCalls:
    """Tests for MiddlewareChain tool call hooks."""

    async def test_before_tool_call_order(self) -> None:
        """Test before_tool_call executes in order."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        result = await chain.before_tool_call("test_tool", {"arg": "value"}, None)

        assert result == {"arg": "value", "mw1": True, "mw2": True}
        assert TrackingMiddleware.call_order == [
            "mw1:before_tool",
            "mw2:before_tool",
        ]

    async def test_after_tool_call_reverse_order(self) -> None:
        """Test after_tool_call executes in reverse order."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        result = await chain.after_tool_call("test_tool", {"arg": "value"}, "result", None)

        # Reverse order: mw2 -> mw1
        assert result == "[mw1][mw2]result"
        assert TrackingMiddleware.call_order == [
            "mw2:after_tool",
            "mw1:after_tool",
        ]


class TestMiddlewareChainOnError:
    """Tests for MiddlewareChain.on_error()."""

    async def test_on_error_empty_chain(self) -> None:
        """Test on_error with empty chain returns None."""
        chain: MiddlewareChain[None] = MiddlewareChain()
        error = ValueError("test")
        result = await chain.on_error(error, None)
        assert result is None

    async def test_on_error_no_handler(self) -> None:
        """Test on_error when no middleware handles error."""
        mw = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw])

        error = ValueError("test")
        result = await chain.on_error(error, None)

        assert result is None
        assert mw.on_error_called

    async def test_on_error_first_handler_wins(self) -> None:
        """Test on_error stops at first handler."""
        mw1 = ErrorHandlerMiddleware("mw1")
        mw2 = ErrorHandlerMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        error = ValueError("original")
        result = await chain.on_error(error, None)

        assert result is not None
        assert "mw1" in str(result)
        # mw2 should not be called since mw1 handled it


class TestMiddlewareChainNested:
    """Tests for nested MiddlewareChain scenarios."""

    async def test_nested_chains_execution_order(self) -> None:
        """Test that nested chains maintain correct execution order."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        inner_chain = MiddlewareChain([mw1, mw2])

        mw3 = TrackingMiddleware("mw3")
        outer_chain = MiddlewareChain([inner_chain, mw3])

        result = await outer_chain.before_run("test", None)

        # Should execute in flattened order: mw1 -> mw2 -> mw3
        assert result == "[mw3][mw2][mw1]test"
        assert TrackingMiddleware.call_order == [
            "mw1:before_run",
            "mw2:before_run",
            "mw3:before_run",
        ]

    async def test_deeply_nested_chains(self) -> None:
        """Test deeply nested chains are properly flattened."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")

        level3 = MiddlewareChain([mw1])
        level2 = MiddlewareChain([level3, mw2])
        level1 = MiddlewareChain([level2, mw3])

        assert len(level1) == 3
        assert level1.middleware == [mw1, mw2, mw3]


class TestMiddlewareChainMethodChaining:
    """Tests for method chaining support."""

    def test_fluent_api(self) -> None:
        """Test fluent API for building chains."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")
        mw4 = TrackingMiddleware("mw4")

        chain: MiddlewareChain[None] = MiddlewareChain().add(mw1).add(mw2).insert(0, mw3).add(mw4)

        assert chain.middleware == [mw3, mw1, mw2, mw4]

    def test_replace_returns_self(self) -> None:
        """Test that replace returns self for chaining."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")

        chain = MiddlewareChain([mw1, mw2]).replace(mw2, mw3).add(mw2)

        assert chain.middleware == [mw1, mw3, mw2]


class TestMiddlewareChainMiddlewareProperty:
    """Tests for the middleware property returning a copy."""

    def test_middleware_returns_copy(self) -> None:
        """Test that middleware property returns a copy."""
        mw1 = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw1])

        middleware_list = chain.middleware

        # Modifying returned list doesn't affect chain
        mw2 = TrackingMiddleware("mw2")
        middleware_list.append(mw2)

        assert len(chain) == 1
        assert mw2 not in chain


class TestMiddlewareChainIntegration:
    """Integration tests with MiddlewareAgent."""

    async def test_chain_with_middleware_agent(self) -> None:
        """Test MiddlewareChain works correctly with MiddlewareAgent."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from pydantic_ai_middleware import MiddlewareAgent

        # Create middleware instances
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")

        # Create a chain with mw1 and mw2
        chain = MiddlewareChain([mw1, mw2])

        # Use chain with standalone middleware mw3
        base_agent = Agent(model=TestModel())
        middleware_agent = MiddlewareAgent(agent=base_agent, middleware=[chain, mw3])

        # Run the agent
        await middleware_agent.run("test prompt")

        # Verify execution order: before_run should be mw1 -> mw2 -> mw3
        # after_run should be mw3 -> mw2 -> mw1
        before_runs = [call for call in TrackingMiddleware.call_order if "before_run" in call]
        assert before_runs == ["mw1:before_run", "mw2:before_run", "mw3:before_run"]

        after_runs = [call for call in TrackingMiddleware.call_order if "after_run" in call]
        assert after_runs == ["mw3:after_run", "mw2:after_run", "mw1:after_run"]

    async def test_nested_chains_with_middleware_agent(self) -> None:
        """Test nested chains with MiddlewareAgent maintain correct order."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from pydantic_ai_middleware import MiddlewareAgent

        # Create nested chain structure
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        inner_chain = MiddlewareChain([mw1, mw2])

        mw3 = TrackingMiddleware("mw3")
        mw4 = TrackingMiddleware("mw4")
        outer_chain = MiddlewareChain([inner_chain, mw3])

        # Use with agent
        base_agent = Agent(model=TestModel())
        middleware_agent = MiddlewareAgent(agent=base_agent, middleware=[outer_chain, mw4])

        # Run the agent
        await middleware_agent.run("test prompt")

        # Verify execution order - should be flattened: mw1 -> mw2 -> mw3 -> mw4
        before_runs = [call for call in TrackingMiddleware.call_order if "before_run" in call]
        assert before_runs == [
            "mw1:before_run",
            "mw2:before_run",
            "mw3:before_run",
            "mw4:before_run",
        ]

        # After runs should be reversed
        after_runs = [call for call in TrackingMiddleware.call_order if "after_run" in call]
        assert after_runs == [
            "mw4:after_run",
            "mw3:after_run",
            "mw2:after_run",
            "mw1:after_run",
        ]

    async def test_chain_as_single_middleware(self) -> None:
        """Test MiddlewareChain can be used as a single middleware object."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from pydantic_ai_middleware import MiddlewareAgent

        # Create a chain
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        chain = MiddlewareChain([mw1, mw2])

        # Use the chain itself as a single middleware (not expanded)
        base_agent = Agent(model=TestModel())
        middleware_agent = MiddlewareAgent(agent=base_agent, middleware=[chain])

        # Run the agent
        await middleware_agent.run("test prompt")

        # Both middleware should have been called through the chain
        assert mw1.before_run_called
        assert mw2.before_run_called
        assert mw1.after_run_called
        assert mw2.after_run_called

        # Verify order
        before_runs = [call for call in TrackingMiddleware.call_order if "before_run" in call]
        assert before_runs == ["mw1:before_run", "mw2:before_run"]


class TestMiddlewareChainErrors:
    """Test error handling in MiddlewareChain."""

    def test_flatten_invalid_item_type(self):
        """Flattening with invalid item type raises error."""
        with pytest.raises(
            TypeError, match="MiddlewareChain items must be AgentMiddleware or MiddlewareChain"
        ):
            MiddlewareChain(["not a middleware"])  # type: ignore

    def test_add_invalid_type(self):
        """Adding invalid type raises error."""
        chain = MiddlewareChain()
        with pytest.raises(
            TypeError, match="MiddlewareChain.add expects AgentMiddleware or MiddlewareChain"
        ):
            chain.add("not a middleware")  # type: ignore

    def test_insert_invalid_type(self):
        """Inserting invalid type raises error."""
        chain = MiddlewareChain()
        with pytest.raises(
            TypeError, match="MiddlewareChain.insert expects AgentMiddleware or MiddlewareChain"
        ):
            chain.insert(0, "not a middleware")  # type: ignore

    def test_replace_with_chain(self):
        """Replace middleware with a chain."""
        mw1 = TrackingMiddleware("mw1")
        mw2 = TrackingMiddleware("mw2")
        mw3 = TrackingMiddleware("mw3")

        chain = MiddlewareChain([mw1, mw2])
        replacement_chain = MiddlewareChain([mw3])

        chain.replace(mw1, replacement_chain)

        assert len(chain) == 2
        assert chain[0] == mw3
        assert chain[1] == mw2

    def test_replace_invalid_type(self):
        """Replacing with invalid type raises error."""
        mw1 = TrackingMiddleware("mw1")
        chain = MiddlewareChain([mw1])

        with pytest.raises(
            TypeError, match="MiddlewareChain.replace expects AgentMiddleware or MiddlewareChain"
        ):
            chain.replace(mw1, "not a middleware")  # type: ignore

    def test_add_operator_invalid_type(self):
        """Adding with + operator with invalid type raises error."""
        chain = MiddlewareChain()

        with pytest.raises(
            TypeError, match="MiddlewareChain \\+ expects AgentMiddleware or MiddlewareChain"
        ):
            chain + "not a middleware"  # type: ignore

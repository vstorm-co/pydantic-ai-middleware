"""Interactive demo: Composable middleware chains.

Run with: uv run python examples/middleware_chains.py

This demo shows how to use MiddlewareChain to:
- Create reusable middleware pipelines
- Compose chains together (chain of chains)
- Modify chains at runtime
- Maintain clear execution order guarantees
"""

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareAgent,
    MiddlewareChain,
    ScopedContext,
)


# Define some example middleware components
class LoggingMiddleware(AgentMiddleware[None]):
    """Logs all interactions."""

    def __init__(self, name: str = "Logger") -> None:
        self.name = name

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        print(f"  [{self.name}] 📥 Input: {prompt}")
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        print(f"  [{self.name}] 📤 Output: {output}")
        return output


class InputSanitizer(AgentMiddleware[None]):
    """Sanitizes and normalizes input."""

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        prompt_str = str(prompt)
        sanitized = prompt_str.strip().lower()
        print(f"  [Sanitizer] 🧹 Cleaned: '{prompt_str}' → '{sanitized}'")
        return sanitized


class ProfanityFilter(AgentMiddleware[None]):
    """Filters profanity from input and output."""

    def __init__(self) -> None:
        self.bad_words = ["bad", "naughty"]

    def _filter(self, text: str) -> str:
        for word in self.bad_words:
            if word in text.lower():
                text = text.replace(word, "***")
        return text

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        filtered = self._filter(str(prompt))
        if filtered != str(prompt):
            print("  [ProfanityFilter] 🚫 Filtered input")
        return filtered

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        filtered = self._filter(str(output))
        if filtered != str(output):
            print("  [ProfanityFilter] 🚫 Filtered output")
        return filtered


class MetricsCollector(AgentMiddleware[None]):
    """Collects metrics about agent execution."""

    def __init__(self) -> None:
        self.run_count = 0

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        self.run_count += 1
        print(f"  [Metrics] 📊 Run #{self.run_count}")
        return prompt

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        print(f"  [Metrics] ✅ Run #{self.run_count} complete")
        return output


class OutputFormatter(AgentMiddleware[None]):
    """Formats output with fancy borders."""

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        formatted = f"✨ {output} ✨"
        print("  [Formatter] 💅 Formatted output")
        return formatted


async def demo_basic_chain() -> None:
    """Demo 1: Basic middleware chain composition."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Middleware Chain")
    print("=" * 70)
    print("\nCreating a reusable validation chain...")

    # Create a reusable validation chain
    validation_chain = MiddlewareChain(
        [
            InputSanitizer(),
            ProfanityFilter(),
        ],
        name="ValidationChain",
    )

    print(f"Chain created: {validation_chain}")
    print(f"Contains {len(validation_chain)} middleware components")

    # Use the chain with an agent
    base_agent = Agent(model=TestModel())
    middleware_agent = MiddlewareAgent(
        agent=base_agent,
        middleware=[
            LoggingMiddleware("PreValidation"),
            validation_chain,  # Chain is used as a single middleware
            LoggingMiddleware("PostValidation"),
        ],
    )

    print("\nRunning agent with validation chain...")
    result = await middleware_agent.run("  Hello WORLD  ")
    print(f"\nFinal result: {result.output}\n")


async def demo_chain_of_chains() -> None:
    """Demo 2: Composing chains together (chain of chains)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Chain of Chains")
    print("=" * 70)
    print("\nBuilding a complex pipeline from smaller chains...")

    # Create specialized chains
    input_chain = MiddlewareChain(
        [InputSanitizer(), ProfanityFilter()],
        name="InputProcessing",
    )

    output_chain = MiddlewareChain(
        [OutputFormatter()],
        name="OutputProcessing",
    )

    monitoring_chain = MiddlewareChain(
        [LoggingMiddleware(), MetricsCollector()],
        name="Monitoring",
    )

    # Compose them into a complete pipeline
    complete_pipeline = MiddlewareChain(
        [
            monitoring_chain,  # Nested chain 1
            input_chain,  # Nested chain 2
            output_chain,  # Nested chain 3
        ],
        name="CompletePipeline",
    )

    print(f"\nPipeline structure: {complete_pipeline}")
    print(f"Total middleware (after flattening): {len(complete_pipeline)}")
    print("\nMiddleware order:")
    for i, mw in enumerate(complete_pipeline, 1):
        print(f"  {i}. {type(mw).__name__}")

    # Use with agent
    base_agent = Agent(model=TestModel())
    middleware_agent = MiddlewareAgent(agent=base_agent, middleware=[complete_pipeline])

    print("\nRunning agent with complete pipeline...")
    result = await middleware_agent.run("test")
    print(f"\nFinal result: {result.output}\n")


async def demo_runtime_modification() -> None:
    """Demo 3: Modifying chains at runtime."""
    print("\n" + "=" * 70)
    print("DEMO 3: Runtime Chain Modification")
    print("=" * 70)
    print("\nStarting with a basic chain and modifying it dynamically...")

    # Start with a basic chain
    chain = MiddlewareChain(
        [InputSanitizer()],
        name="DynamicChain",
    )

    print(f"\nInitial chain: {chain}")
    print(f"Middleware: {[type(mw).__name__ for mw in chain]}")

    # Add middleware
    print("\n➕ Adding ProfanityFilter...")
    chain.add(ProfanityFilter())
    print(f"Chain now: {[type(mw).__name__ for mw in chain]}")

    # Insert at beginning
    print("\n➕ Inserting LoggingMiddleware at beginning...")
    chain.insert(0, LoggingMiddleware("Start"))
    print(f"Chain now: {[type(mw).__name__ for mw in chain]}")

    # Add at end
    print("\n➕ Adding LoggingMiddleware at end...")
    chain.add(LoggingMiddleware("End"))
    print(f"Chain now: {[type(mw).__name__ for mw in chain]}")

    # Use fluent API
    print("\n➕ Using fluent API to add MetricsCollector and OutputFormatter...")
    chain.add(MetricsCollector()).add(OutputFormatter())
    print(f"Chain now: {[type(mw).__name__ for mw in chain]}")

    # Test the modified chain
    base_agent = Agent(model=TestModel())
    middleware_agent = MiddlewareAgent(agent=base_agent, middleware=[chain])

    print("\nRunning agent with modified chain...")
    result = await middleware_agent.run("Hello")
    print(f"\nFinal result: {result.output}")

    # Remove middleware
    print("\n➖ Removing OutputFormatter...")
    formatter = chain.pop()
    print(f"Removed: {type(formatter).__name__}")
    print(f"Chain now: {[type(mw).__name__ for mw in chain]}")

    # Run again
    print("\nRunning again after removal...")
    result = await middleware_agent.run("Hello again")
    print(f"\nFinal result: {result.output}\n")


async def demo_operator_overloads() -> None:
    """Demo 4: Using operator overloads for composition."""
    print("\n" + "=" * 70)
    print("DEMO 4: Operator Overloads (+, +=)")
    print("=" * 70)

    # Create two chains
    chain1 = MiddlewareChain([InputSanitizer(), ProfanityFilter()], name="Chain1")
    chain2 = MiddlewareChain([MetricsCollector(), OutputFormatter()], name="Chain2")

    print(f"\nChain1: {[type(mw).__name__ for mw in chain1]}")
    print(f"Chain2: {[type(mw).__name__ for mw in chain2]}")

    # Combine with + operator (creates new chain)
    print("\n➕ Combining with + operator...")
    combined = chain1 + chain2
    print(f"Combined: {[type(mw).__name__ for mw in combined]}")
    print(f"Original chain1 unchanged: {len(chain1)} middleware")
    print(f"Original chain2 unchanged: {len(chain2)} middleware")

    # Use += operator (modifies in place)
    print("\n➕ Using += operator on chain1...")
    chain1 += LoggingMiddleware()
    print(f"Chain1 now: {[type(mw).__name__ for mw in chain1]}")

    # Add single middleware with +
    print("\n➕ Adding single middleware with +...")
    extended = combined + LoggingMiddleware("Extra")
    print(f"Extended: {[type(mw).__name__ for mw in extended]}\n")


async def demo_execution_order() -> None:
    """Demo 5: Execution order guarantees."""
    print("\n" + "=" * 70)
    print("DEMO 5: Execution Order Guarantees")
    print("=" * 70)
    print("\nDemonstrating before_* (forward) and after_* (reverse) order...\n")

    # Create middleware that shows execution order clearly
    class OrderTracker(AgentMiddleware[None]):
        def __init__(self, position: int) -> None:
            self.position = position

        async def before_run(
            self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
        ) -> str | Sequence[Any]:
            print(f"  before_run #{self.position}")
            return prompt

        async def after_run(
            self,
            prompt: str | Sequence[Any],
            output: Any,
            deps: None,
            ctx: ScopedContext | None = None,
        ) -> Any:
            print(f"  after_run #{self.position}")
            return output

    # Create chain with numbered middleware
    chain = MiddlewareChain(
        [OrderTracker(1), OrderTracker(2), OrderTracker(3)],
        name="OrderDemo",
    )

    base_agent = Agent(model=TestModel())
    middleware_agent = MiddlewareAgent(agent=base_agent, middleware=[chain])

    print("Expected order:")
    print("  before_run: 1 → 2 → 3")
    print("  after_run:  3 → 2 → 1")
    print("\nActual execution:")

    await middleware_agent.run("test")
    print()


async def main() -> None:
    """Run all demos."""
    print("\n🚀 MiddlewareChain Demo Suite")
    print("=" * 70)

    await demo_basic_chain()
    await demo_chain_of_chains()
    await demo_runtime_modification()
    await demo_operator_overloads()
    await demo_execution_order()

    print("\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

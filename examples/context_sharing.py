"""Interactive demo: Context sharing between middleware hooks.

Run with: uv run python examples/context_sharing.py

This demo shows how middleware can share data through the context system:
- Each hook writes data to its namespace
- Later hooks can read from earlier hooks
- The demo shows this data flow in real-time
"""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    HookType,
    InputBlocked,
    MiddlewareAgent,
    MiddlewareContext,
    ScopedContext,
)


class InputAnalyzer(AgentMiddleware[None]):
    """Analyzes input and stores metrics in context.

    Behavior changes based on config:
    - config['strict_mode']: blocks short prompts
    - config['max_length']: truncates long prompts
    """

    async def before_run(
        self, prompt: str | Sequence[Any], deps: None, ctx: ScopedContext | None = None
    ) -> str | Sequence[Any]:
        prompt_str = str(prompt)

        if ctx:
            # Store analysis in context
            ctx.set("input_length", len(prompt_str))
            ctx.set("word_count", len(prompt_str.split()))
            ctx.set("analyzed_at", time.time())

            # Config-driven behavior
            strict = ctx.config.get("strict_mode", False)
            max_len = ctx.config.get("max_length", 200)

            if strict and len(prompt_str) < 10:
                ctx.set("blocked", True)
                ctx.set("block_reason", "too_short")
                raise InputBlocked("Prompt too short (strict mode)")

            if len(prompt_str) > max_len:
                ctx.set("truncated", True)
                ctx.set("original_length", len(prompt_str))
                return prompt_str[:max_len] + "..."

            ctx.set("blocked", False)
            ctx.set("truncated", False)

        return prompt


class ResponseEnhancer(AgentMiddleware[None]):
    """Enhances response based on data from earlier hooks.

    Reads from BEFORE_RUN to make decisions.
    """

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        if ctx:
            # Read from BEFORE_RUN hook
            word_count = ctx.get_from(HookType.BEFORE_RUN, "word_count", 0)
            truncated = ctx.get_from(HookType.BEFORE_RUN, "truncated", False)

            # Store processing info
            ctx.set("response_length", len(str(output)))
            ctx.set(
                "processing_time_ms",
                (time.time() - ctx.get_from(HookType.BEFORE_RUN, "analyzed_at", 0)) * 1000,
            )

            # Enhance based on context
            suffix = ""
            if truncated:
                suffix = " [Note: Your input was truncated]"
            elif word_count == 1:
                suffix = " [Tip: Try longer prompts for better results]"

            return str(output) + suffix

        return output


class MetricsCollector(AgentMiddleware[None]):
    """Collects metrics from all hooks into a summary.

    Demonstrates reading from multiple earlier hooks.
    """

    async def after_run(
        self, prompt: str | Sequence[Any], output: Any, deps: None, ctx: ScopedContext | None = None
    ) -> Any:
        if ctx:
            # Collect from all earlier hooks
            metrics = {
                "input": {
                    "length": ctx.get_from(HookType.BEFORE_RUN, "input_length"),
                    "words": ctx.get_from(HookType.BEFORE_RUN, "word_count"),
                    "truncated": ctx.get_from(HookType.BEFORE_RUN, "truncated"),
                },
                "output": {
                    "length": ctx.get_from(HookType.AFTER_RUN, "response_length"),
                },
                "timing_ms": ctx.get_from(HookType.AFTER_RUN, "processing_time_ms"),
            }
            ctx.set("metrics_summary", metrics)

        return output


# =============================================================================
# Demo helpers
# =============================================================================


def create_model() -> TestModel:
    model = TestModel()
    model.custom_output_text = "Hello! I'm here to help."
    return model


def create_agent(config: dict[str, Any]) -> tuple[MiddlewareAgent[None, str], MiddlewareContext]:
    """Create agent with context and middleware."""
    ctx = MiddlewareContext(config=config)

    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[
            InputAnalyzer(),
            ResponseEnhancer(),
            MetricsCollector(),
        ],
        context=ctx,
    )

    return agent, ctx


async def run_with_context(prompt: str, config: dict[str, Any]) -> None:
    """Run agent and display context data flow."""
    agent, ctx = create_agent(config)

    print(f"\n  Input: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    strict = config.get("strict_mode", False)
    max_len = config.get("max_length", 200)
    print(f"  Config: strict_mode={strict}, max_length={max_len}")

    try:
        result = await agent.run(prompt)

        # Show data from each hook
        print("\n  Context Data Flow:")

        # BEFORE_RUN data
        br_data = ctx._hook_data[HookType.BEFORE_RUN]
        print(
            f"     BEFORE_RUN wrote: input_length={br_data.get('input_length')}, "
            f"word_count={br_data.get('word_count')}, truncated={br_data.get('truncated')}"
        )

        # AFTER_RUN data
        ar_data = ctx._hook_data[HookType.AFTER_RUN]
        print(
            f"     AFTER_RUN wrote: response_length={ar_data.get('response_length')}, "
            f"processing_time_ms={ar_data.get('processing_time_ms', 0):.1f}"
        )

        # Metrics summary
        metrics = ar_data.get("metrics_summary", {})
        if metrics:
            print(f"     AFTER_RUN metrics: {metrics}")

        print(f"\n  Output: {result.output}")

    except InputBlocked as e:
        print(f"\n  BLOCKED: {e.reason}")
        br_data = ctx._hook_data[HookType.BEFORE_RUN]
        print(f"     BEFORE_RUN wrote: blocked=True, block_reason={br_data.get('block_reason')}")


async def main() -> None:
    print("=" * 60)
    print("CONTEXT SHARING DEMO")
    print("=" * 60)
    print("\nThis demo shows how middleware shares data through context:")
    print("  InputAnalyzer (BEFORE_RUN) -> analyzes & stores input metrics")
    print("  ResponseEnhancer (AFTER_RUN) -> reads BEFORE_RUN data, enhances response")
    print("  MetricsCollector (AFTER_RUN) -> reads all hooks, creates summary")
    print("\nContext features:")
    print("  config: Read-only settings (strict_mode, max_length)")
    print("  Hook namespaces: Each hook writes to its own, reads from earlier")
    print("\nCommands:")
    print("  /strict on|off  - Toggle strict mode (blocks short prompts)")
    print("  /maxlen N       - Set max prompt length")
    print("  /show           - Show current config")
    print("  /quit           - Exit")
    print()

    config: dict[str, Any] = {"strict_mode": False, "max_length": 100}

    while True:
        try:
            mode = "STRICT" if config.get("strict_mode") else "NORMAL"
            maxlen = config.get("max_length", 100)
            prompt = input(f"[{mode} | max={maxlen}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "/quit":
            print("Bye!")
            break

        if prompt.lower() == "/strict on":
            config["strict_mode"] = True
            print("Strict mode enabled (short prompts will be blocked)\n")
            continue

        if prompt.lower() == "/strict off":
            config["strict_mode"] = False
            print("Strict mode disabled\n")
            continue

        if prompt.lower().startswith("/maxlen "):
            try:
                config["max_length"] = int(prompt.split()[1])
                print(f"Max length set to {config['max_length']}\n")
            except (ValueError, IndexError):
                print("Usage: /maxlen N (where N is a number)\n")
            continue

        if prompt.lower() == "/show":
            print("\nCurrent config:")
            print(f"  strict_mode: {config.get('strict_mode', False)}")
            print(f"  max_length: {config.get('max_length', 100)}\n")
            continue

        if prompt.startswith("/"):
            print(f"Unknown command: {prompt}")
            print("Commands: /strict on|off, /maxlen N, /show, /quit\n")
            continue

        # Run with the prompt
        await run_with_context(prompt, config)
        print()


if __name__ == "__main__":
    asyncio.run(main())

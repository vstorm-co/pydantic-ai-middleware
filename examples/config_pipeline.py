"""Example: load middleware pipeline config and run an agent.

Usage:
  `uv run python examples/config_pipeline.py --role admin`
  `uv run python examples/config_pipeline.py --role user`

This example exercises all pipeline skeleton nodes:
- `type`: instantiate registered middleware
- `chain`: sequential middleware
- `parallel`: concurrent middleware (using ParallelMiddleware)
- `when`: conditional branching using a registered predicate
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_middleware.base import AgentMiddleware
from pydantic_ai_middleware.builder import MiddlewareRegistry
from pydantic_ai_middleware.config_loaders import load_middleware_config_path
from pydantic_ai_middleware.context import HookType, MiddlewareContext, ScopedContext


class TagOutputMiddleware(AgentMiddleware[None]):
    def __init__(self, *, tag: str) -> None:
        self.tag = tag

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        return f"[{self.tag}]{output}"


class PrefixPromptMiddleware(AgentMiddleware[None]):
    def __init__(self, *, prefix: str) -> None:
        self.prefix = prefix

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        if isinstance(prompt, str):
            return f"{self.prefix}{prompt}"
        return prompt


class LogToContextMiddleware(AgentMiddleware[None]):
    def __init__(self, *, key: str) -> None:
        self.key = key

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        if ctx is not None:
            ctx.set(self.key, prompt)
        return prompt


async def main(role: str) -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry()

    @reg.middleware_factory("tag_output")
    def make_tag_output(*, tag: str) -> TagOutputMiddleware:
        return TagOutputMiddleware(tag=tag)

    @reg.middleware_factory("prefix_prompt")
    def make_prefix_prompt(*, prefix: str) -> PrefixPromptMiddleware:
        return PrefixPromptMiddleware(prefix=prefix)

    @reg.middleware_factory("log_to_ctx")
    def make_log_to_ctx(*, key: str) -> LogToContextMiddleware:
        return LogToContextMiddleware(key=key)

    @reg.predicate("is_admin")
    def is_admin(ctx: ScopedContext | None) -> bool:
        if ctx is None:
            return False
        return ctx.config.get("role") == "admin"

    pipeline = load_middleware_config_path(
        Path("examples/pipeline.yaml"),
        registry=reg.middleware,
        predicates=reg.predicates,
    )

    model = TestModel()
    model.custom_output_text = "hello"
    base_agent = Agent(model=model, output_type=str)
    ctx = MiddlewareContext(config={"role": role})

    agent = MiddlewareAgent(agent=base_agent, middleware=pipeline, context=ctx)
    result = await agent.run("test prompt")

    print("Output:", result.output)
    before_run_ctx = ctx.for_hook(HookType.BEFORE_RUN)
    print("Context(before_run):", before_run_ctx.get_all_from(HookType.BEFORE_RUN))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the config pipeline example.")
    parser.add_argument(
        "--role",
        default="admin",
        choices=["admin", "user"],
        help="Role used by the `is_admin` predicate for the `when` node.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.role))

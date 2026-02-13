"""Example: build a pipeline in Python and export it to JSON/YAML.

Run:
  `uv run python examples/spec_build_and_export.py`

This demonstrates:
  - defining a portable pipeline spec in Python (PipelineSpec)
  - exporting to `examples/pipeline.from_spec.json` / `.yaml`
  - compiling into middleware instances and running an agent
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_middleware.base import AgentMiddleware
from pydantic_ai_middleware.builder import MiddlewarePipelineCompiler, MiddlewareRegistry
from pydantic_ai_middleware.context import ScopedContext
from pydantic_ai_middleware.pipeline_spec import (
    PipelineSpec,
    chain_node,
    parallel_node,
    type_node,
    when_node,
)
from pydantic_ai_middleware.strategies import AggregationStrategy


class TagOutputMiddleware(AgentMiddleware[None]):
    def __init__(self, *, tag: str) -> None:
        self.tag = tag

    async def after_run(
        self,
        prompt: str,
        output: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        return f"[{self.tag}]{output}"


async def main() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry()

    @reg.middleware_factory("tag_output")
    def make_tag_output(*, tag: str) -> TagOutputMiddleware:
        return TagOutputMiddleware(tag=tag)

    @reg.predicate("always_true")
    def always_true(ctx: ScopedContext | None) -> bool:
        return True

    spec = (
        PipelineSpec()
        .add(
            parallel_node(
                [
                    type_node("tag_output", {"tag": "P1"}),
                    type_node("tag_output", {"tag": "P2"}),
                ],
                strategy=AggregationStrategy.ALL_MUST_PASS,
            )
        )
        .add(
            when_node(
                predicate="always_true",
                then=[type_node("tag_output", {"tag": "THEN"})],
                else_=[type_node("tag_output", {"tag": "ELSE"})],
            )
        )
        .add(chain_node([type_node("tag_output", {"tag": "END"})]))
    )

    spec.save("examples/pipeline_from_spec.json")
    spec.save("examples/pipeline_from_spec.yaml")

    compiler = MiddlewarePipelineCompiler(registry=reg)
    middleware = spec.compile(compiler)

    model = TestModel()
    model.custom_output_text = "hello"
    agent = MiddlewareAgent(agent=Agent(model=model, output_type=str), middleware=middleware)
    result = await agent.run("test")

    print("Saved:")
    print("-", Path("examples/pipeline_from_spec.json"))
    print("-", Path("examples/pipeline_from_spec.yaml"))
    print("Output:", result.output)


if __name__ == "__main__":
    asyncio.run(main())

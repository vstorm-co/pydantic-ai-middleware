"""Example: Decorator-based middleware — lightweight alternative to classes.

Instead of subclassing AgentMiddleware, use decorators:
  @before_run, @after_run, @before_tool_call, @after_tool_call,
  @before_model_request, @on_error, @on_tool_error

Run: uv run python examples/hooks_showcase/12_decorator_middleware.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_middleware.decorators import (
    after_run,
    after_tool_call,
    before_model_request,
    before_run,
    before_tool_call,
)


@before_run
async def log_prompt(
    prompt: str | Sequence[Any],
    deps: None,
    ctx: Any = None,
) -> str | Sequence[Any]:
    print(f"[decorator:before_run] {prompt!r}")
    return prompt


@after_run
async def log_output(
    prompt: str | Sequence[Any],
    output: Any,
    deps: None,
    ctx: Any = None,
) -> Any:
    print(f"[decorator:after_run] {output!r}")
    return output


@before_model_request
async def log_messages(
    messages: list[ModelMessage],
    deps: None,
    ctx: Any = None,
) -> list[ModelMessage]:
    print(f"[decorator:before_model_request] {len(messages)} messages")
    return messages


@before_tool_call
async def log_tool_start(
    tool_name: str,
    tool_args: dict[str, Any],
    deps: None,
    ctx: Any = None,
) -> dict[str, Any]:
    print(f"[decorator:before_tool_call] {tool_name}({tool_args})")
    return tool_args


@after_tool_call
async def log_tool_result(
    tool_name: str,
    tool_args: dict[str, Any],
    result: Any,
    deps: None,
    ctx: Any = None,
) -> Any:
    print(f"[decorator:after_tool_call] {tool_name} -> {result!r}")
    return result


agent = Agent(
    "openai:gpt-4.1",
    instructions="Use the greet tool to answer. Be brief.",
)

toolset: FunctionToolset[None] = FunctionToolset()


@toolset.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


async def main() -> None:
    mw_agent = MiddlewareAgent(
        agent=agent,
        middleware=[log_prompt, log_output, log_messages, log_tool_start, log_tool_result],
    )

    result = await mw_agent.run("Greet Alice", toolsets=[toolset])
    print(f"\n[result] {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

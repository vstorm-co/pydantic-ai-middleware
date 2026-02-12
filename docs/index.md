<h1 align="center">Pydantic AI Middleware</h1>
<p align="center">
  <em>Simple middleware for Pydantic AI agents, the Pythonic way</em>
</p>
<p align="center">
  <a href="https://github.com/vstorm-co/pydantic-ai-middleware/actions/workflows/ci.yml"><img src="https://github.com/vstorm-co/pydantic-ai-middleware/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/vstorm-co/pydantic-ai-middleware"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage"></a>
  <a href="https://pypi.org/project/pydantic-ai-middleware/"><img src="https://img.shields.io/pypi/v/pydantic-ai-middleware.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

**Pydantic AI Middleware** is a lightweight library for adding before/after hooks to [Pydantic AI](https://ai.pydantic.dev/) agents. No imposed structure — you decide what to build: logging, guardrails, metrics, rate limiting, PII redaction, or anything else.

Part of the [vstorm-co](https://github.com/vstorm-co) ecosystem for building production AI agents with Pydantic AI.

## Why use Pydantic AI Middleware?

1. **Clean API**: Simple before/after hooks at 6 lifecycle stages — no complex abstractions to learn.

2. **Maximum Flexibility**: No imposed guardrail structure. You decide what each hook does.

3. **Production Ready**: 100% test coverage, strict typing with Pyright + MyPy, and parallel execution with early cancellation.

4. **Composable**: Chain, branch, parallelize, and load middleware from config files.

## Hello World Example

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware, InputBlocked

class SecurityMiddleware(AgentMiddleware[None]):
    """Block dangerous inputs before they reach the agent."""

    async def before_run(self, prompt, deps, ctx):
        if "dangerous" in prompt.lower():
            raise InputBlocked("Dangerous content detected")
        return prompt

class LoggingMiddleware(AgentMiddleware[None]):
    """Log agent activity."""

    async def before_run(self, prompt, deps, ctx):
        print(f"Starting: {prompt[:50]}...")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        print(f"Finished: {output}")
        return output

agent = MiddlewareAgent(
    agent=Agent('openai:gpt-4o'),
    middleware=[LoggingMiddleware(), SecurityMiddleware()],
)

result = await agent.run("Hello, how are you?")
```

## Decorator Syntax

For simple middleware, use decorators:

```python
from pydantic_ai_middleware import before_run, after_run, ToolBlocked, before_tool_call

@before_run
async def log_input(prompt, deps, ctx):
    print(f"Input: {prompt}")
    return prompt

@before_tool_call
async def block_dangerous_tools(tool_name, tool_args, deps, ctx):
    if tool_name == "delete_file":
        raise ToolBlocked(tool_name, "Not allowed")
    return tool_args
```

## Core Capabilities

| Capability | Description |
|------------|-------------|
| **6 Lifecycle Hooks** | before_run, after_run, before_model_request, before_tool_call, after_tool_call, on_error |
| **Parallel Execution** | Run multiple middleware concurrently with 4 aggregation strategies |
| **Async Guardrails** | Run guardrails alongside LLM calls (BLOCKING, CONCURRENT, ASYNC_POST) |
| **Middleware Chains** | Compose middleware into reusable sequences with `+` operator |
| **Conditional Routing** | Route to different middleware based on runtime conditions |
| **Config Loading** | Build pipelines from JSON/YAML configuration files |
| **Context Sharing** | Share data between hooks with access control |
| **Decorator Syntax** | Create middleware from simple decorated functions |

## Part of the Ecosystem

Pydantic AI Middleware works alongside other vstorm-co packages:

| Package | Description |
|---------|-------------|
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation: Agent framework by Pydantic |
| [pydantic-deep](https://github.com/vstorm-co/pydantic-deep) | Full agent framework with planning, subagents, skills |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage, Docker sandbox, permission controls |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning with PostgreSQL and event streaming |
| [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) | Multi-agent orchestration |
| [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) | Context management processors |

## Installation

```bash
pip install pydantic-ai-middleware
```

With YAML config support:

```bash
pip install pydantic-ai-middleware[yaml]
```

## llms.txt

Pydantic AI Middleware supports the [llms.txt](https://llmstxt.org/) standard. Access documentation at `/llms.txt` for LLM-optimized content.

## Next Steps

- [Installation](installation.md) - Get started in minutes
- [Core Concepts](concepts/index.md) - Learn about middleware, hooks, and context
- [Advanced Features](advanced/middleware-chains.md) - Chains, parallel execution, config loading
- [Examples](examples/index.md) - Real-world examples
- [API Reference](api/index.md) - Complete API documentation

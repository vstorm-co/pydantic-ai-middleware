<h1 align="center">Pydantic AI Middleware</h1>

<p align="center">
  <b>Clean Before/After Hooks for Pydantic AI Agents — No Imposed Structure</b>
</p>

<p align="center">
  <a href="https://vstorm-co.github.io/pydantic-ai-middleware/">Docs</a> •
  <a href="https://vstorm-co.github.io/pydantic-ai-middleware/examples/">Examples</a> •
  <a href="https://pypi.org/project/pydantic-ai-middleware/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/pydantic-ai-middleware/"><img src="https://img.shields.io/pypi/v/pydantic-ai-middleware.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/vstorm-co/pydantic-ai-middleware"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen.svg" alt="Coverage"></a>
  <a href="https://github.com/vstorm-co/pydantic-ai-middleware/actions/workflows/ci.yml"><img src="https://github.com/vstorm-co/pydantic-ai-middleware/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

<p align="center">
  <b>7 Lifecycle Hooks</b>
  &nbsp;&bull;&nbsp;
  <b>Parallel Execution</b>
  &nbsp;&bull;&nbsp;
  <b>Async Guardrails</b>
  &nbsp;&bull;&nbsp;
  <b>Fully Type-Safe</b>
</p>

---

## Get Started in 60 Seconds

```bash
pip install pydantic-ai-middleware
```

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware, InputBlocked

class ContentFilter(AgentMiddleware[None]):
    """Block dangerous prompts before they reach the LLM."""

    async def before_run(self, prompt, deps, ctx=None):
        if "ignore all instructions" in prompt.lower():
            raise InputBlocked("Prompt injection attempt blocked")
        return prompt

    async def after_run(self, prompt, output, deps, ctx=None):
        # Redact sensitive patterns from the LLM response
        return output.replace("SSN:", "[REDACTED]")

# Wrap any pydantic-ai Agent with middleware
base_agent = Agent("openai:gpt-4o", instructions="You are a helpful assistant.")
agent = MiddlewareAgent(agent=base_agent, middleware=[ContentFilter()])

result = await agent.run("Hello, how are you?")
print(result.output)
```

**That's it.** Your pydantic-ai agent now has input validation, output filtering, and prompt injection protection — all with a simple wrapper.

---

## Why Middleware, Not Guardrails?

pydantic-ai-middleware takes a different approach from traditional guardrails libraries:

| Aspect | Middleware (this library) | Traditional Guardrails |
|--------|---------------------------|------------------------|
| Complexity | Low | High |
| Structure | No imposed structure | Fixed result types, actions |
| Flexibility | Maximum | Constrained by design |
| Learning curve | Flat | Steeper |
| Built-in guardrails | None (you build what you need) | Pre-built (PII, moderation) |
| Parallel execution | Built-in with early cancellation | Often built-in |
| Type safety | Full generics support | Varies |

**You decide what to build.** Logging, guardrails, metrics, rate limiting, PII redaction — all using the same simple API.

---

## Features

**7 Lifecycle Hooks** — before_run, after_run, before_model_request, before_tool_call, on_tool_error, after_tool_call, on_error

**Parallel Execution** — Run multiple middleware concurrently with 4 aggregation strategies and early cancellation

**Async Guardrails** — Run guardrails alongside LLM calls with BLOCKING, CONCURRENT, or ASYNC_POST timing

**Middleware Chains** — Compose middleware into reusable, ordered sequences with `+` operator

**Conditional Routing** — Route to different middleware based on runtime conditions

**Config Loading** — Build pipelines from JSON/YAML configuration files

**Decorator Syntax** — Create middleware from simple decorated functions

**Context Sharing** — Share data between hooks with access control

**Tool Name Filtering** — Scope middleware to specific tools with `tool_names`

**Hook Timeouts** — Per-middleware timeout enforcement with `MiddlewareTimeout`

**Permission Decisions** — Structured ALLOW/DENY/ASK protocol for tool authorization

**Zero Overhead** — No mandatory dependencies beyond pydantic-ai-slim

---

## Hook Lifecycle

```
                          ╭─────────╮
                          │  input  │
                          ╰────┬────╯
                               │
                               ▼
                    ┌────────────────────┐
                    │    before_run      │─── can block (InputBlocked)
                    └─────────┬──────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
             ┌──▶│  before_model_request   │─── modify messages
             │   └────────────┬────────────┘
             │                │
             │                ▼
             │          ╔═══════════╗
             │          ║   Model   ║
             │          ╚═════╤═════╝
             │           ╱         ╲
             │      tool call    finish
             │         ╱             ╲
             │        ▼               ▼
             │  ┌──────────────┐  ┌──────────────┐
             │  │before_tool_  │  │   after_run  │─── modify output
             │  │    call      │  └──────┬───────┘
             │  └──────┬───────┘         │
             │         │            ╭────┴────╮
             │         ▼            │ output  │
             │    ┌─────────┐       ╰─────────╯
             │    │  tool   │
             │    └────┬────┘
             │     ╱       ╲
             │  success   error
             │    │         │
             │    ▼         ▼
             │ ┌────────┐ ┌───────────────┐
             │ │ after_ │ │ on_tool_error │─── replace exception
             │ │ tool_  │ └───────┬───────┘
             │ │  call  │         │
             │ └───┬────┘         │
             │     └──────┬───────┘
             │            │
             └────────────┘
              observation

      ── on_error: called on any unhandled exception ──
```

| Hook | When Called | Can Modify |
|------|-------------|------------|
| `before_run` | Before agent starts | Prompt |
| `after_run` | After agent finishes | Output |
| `before_model_request` | Before each model call | Messages |
| `before_tool_call` | Before tool execution | Tool arguments |
| `on_tool_error` | When a tool raises an exception | Exception (replace or re-raise) |
| `after_tool_call` | After tool execution | Tool result |
| `on_error` | When error occurs | Exception |

---

## Real-World Examples

### Input Validation + Rate Limiting

```python
import time
from pydantic_ai import Agent
from pydantic_ai_middleware import (
    MiddlewareAgent,
    AgentMiddleware,
    InputBlocked,
    MiddlewareContext,
)

class RateLimiter(AgentMiddleware[None]):
    """Limit how many requests per minute."""

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []

    async def before_run(self, prompt, deps, ctx=None):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_per_minute:
            raise InputBlocked("Rate limit exceeded — try again later")
        self._timestamps.append(now)
        return prompt

class PromptSanitizer(AgentMiddleware[None]):
    """Remove potentially harmful instructions from prompts."""

    BLOCKED_PATTERNS = ["ignore previous", "system prompt", "jailbreak"]

    async def before_run(self, prompt, deps, ctx=None):
        lower = prompt.lower() if isinstance(prompt, str) else str(prompt).lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in lower:
                raise InputBlocked(f"Blocked pattern detected: {pattern}")
        return prompt

# Build the agent with middleware pipeline
base_agent = Agent("openai:gpt-4o", instructions="You are a customer support agent.")

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[RateLimiter(max_per_minute=20), PromptSanitizer()],
    context=MiddlewareContext(),  # enable cross-middleware data sharing
)

result = await agent.run("How do I reset my password?")
```

### Tool Authorization with Permission Decisions

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_middleware import (
    MiddlewareAgent,
    AgentMiddleware,
    ToolDecision,
    ToolPermissionResult,
)

# Define a pydantic-ai agent with tools
base_agent = Agent("openai:gpt-4o", instructions="You are a file manager.")

@base_agent.tool
async def read_file(ctx: RunContext[None], path: str) -> str:
    """Read a file from disk."""
    return f"Contents of {path}"

@base_agent.tool
async def delete_file(ctx: RunContext[None], path: str) -> str:
    """Delete a file from disk."""
    return f"Deleted {path}"

# Middleware that controls tool access
class FileAccessControl(AgentMiddleware[None]):
    """Require explicit approval for destructive file operations."""

    tool_names = {"delete_file"}  # only intercept delete_file

    async def before_tool_call(self, tool_name, tool_args, deps, ctx=None):
        return ToolPermissionResult(
            decision=ToolDecision.ASK,
            reason=f"Agent wants to delete: {tool_args.get('path')}",
        )

# Permission handler — called when middleware returns ASK
async def approval_callback(tool_name: str, tool_args: dict, reason: str) -> bool:
    print(f"[APPROVAL REQUIRED] {reason}")
    response = input("Allow? (y/n): ")
    return response.lower() == "y"

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[FileAccessControl()],
    permission_handler=approval_callback,
)

# read_file works without approval, delete_file triggers the callback
result = await agent.run("Read config.yaml then delete temp.log")
```

### Structured Output with Audit Logging

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_middleware import (
    MiddlewareAgent,
    AgentMiddleware,
    MiddlewareContext,
    ScopedContext,
)

class SupportTicket(BaseModel):
    category: str
    priority: str
    summary: str

class AuditLogger(AgentMiddleware[None]):
    """Log all agent interactions for compliance."""

    async def before_run(self, prompt, deps, ctx=None):
        if ctx:
            ctx.set("input_prompt", prompt)
        print(f"[AUDIT] Input: {prompt[:80]}...")
        return prompt

    async def after_run(self, prompt, output, deps, ctx=None):
        print(f"[AUDIT] Output type: {type(output).__name__}")
        return output

    async def before_tool_call(self, tool_name, tool_args, deps, ctx=None):
        print(f"[AUDIT] Tool call: {tool_name}({tool_args})")
        return tool_args

# Agent with structured output — middleware works transparently
base_agent = Agent(
    "openai:gpt-4o",
    instructions="Classify support tickets.",
    output_type=SupportTicket,
)

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[AuditLogger()],
    context=MiddlewareContext(),
)

result = await agent.run("My payment failed and I can't access my account")
ticket: SupportTicket = result.output
print(f"Category: {ticket.category}, Priority: {ticket.priority}")
```

### Parallel Validators with Async Guardrails

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import (
    MiddlewareAgent,
    AgentMiddleware,
    ParallelMiddleware,
    AsyncGuardrailMiddleware,
    AggregationStrategy,
    GuardrailTiming,
    InputBlocked,
)

class ProfanityFilter(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx=None):
        # Simulated check — replace with real classifier
        if any(word in prompt.lower() for word in ["badword"]):
            raise InputBlocked("Profanity detected")
        return prompt

class PIIDetector(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx=None):
        import re
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", prompt):
            raise InputBlocked("SSN detected in input")
        return prompt

class ToxicityChecker(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx=None):
        # Simulated slow ML-based check
        import asyncio
        await asyncio.sleep(0.5)
        return prompt

# Run ProfanityFilter + PIIDetector in parallel (fast, both must pass)
fast_validators = ParallelMiddleware(
    middleware=[ProfanityFilter(), PIIDetector()],
    strategy=AggregationStrategy.ALL_MUST_PASS,
)

# Run ToxicityChecker concurrently with the LLM (saves latency)
toxicity_guard = AsyncGuardrailMiddleware(
    guardrail=ToxicityChecker(),
    timing=GuardrailTiming.CONCURRENT,
    cancel_on_failure=True,
)

base_agent = Agent("openai:gpt-4o")
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[fast_validators, toxicity_guard],
)

result = await agent.run("Summarize this document for me")
```

### Decorator Syntax for Quick Middleware

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import (
    MiddlewareAgent,
    before_run,
    after_run,
    before_tool_call,
    on_tool_error,
)

@before_run
async def log_input(prompt, deps, ctx=None):
    print(f">>> {prompt}")
    return prompt

@after_run
async def log_output(prompt, output, deps, ctx=None):
    print(f"<<< {output}")
    return output

@before_tool_call(tools={"web_search"})
async def validate_search(tool_name, tool_args, deps, ctx=None):
    """Only runs for web_search tool, skipped for all others."""
    query = tool_args.get("query", "")
    if len(query) > 500:
        tool_args["query"] = query[:500]
    return tool_args

@on_tool_error(tools={"web_search"})
async def handle_search_error(tool_name, tool_args, error, deps, ctx=None):
    if isinstance(error, TimeoutError):
        return ConnectionError("Search service temporarily unavailable")
    return None  # re-raise original error

base_agent = Agent("openai:gpt-4o")
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[log_input, log_output, validate_search, handle_search_error],
)

result = await agent.run("Search for the latest Python release notes")
```

### Middleware Chains + Conditional Routing

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import (
    MiddlewareAgent,
    MiddlewareChain,
    ConditionalMiddleware,
    AgentMiddleware,
)

class AuthMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx=None):
        if ctx:
            ctx.set("authenticated", True)
        return prompt

class AdminAudit(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx=None):
        print("[ADMIN AUDIT] Elevated access logged")
        return prompt

class UserAudit(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx=None):
        print("[USER AUDIT] Standard access logged")
        return prompt

# Reusable security chain
security = MiddlewareChain([AuthMiddleware()], name="security")

# Route to different audit middleware based on runtime condition
audit = ConditionalMiddleware(
    condition=lambda ctx: ctx is not None and ctx.get("is_admin", False),
    when_true=AdminAudit(),
    when_false=UserAudit(),
)

# Combine with + operator
pipeline = security + MiddlewareChain([audit])

base_agent = Agent("openai:gpt-4o")
agent = MiddlewareAgent(agent=base_agent, middleware=[pipeline])
```

### Hook Timeouts

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware, MiddlewareTimeout

class ExternalAPICheck(AgentMiddleware[None]):
    timeout = 3.0  # seconds — applies to every hook on this middleware

    async def before_run(self, prompt, deps, ctx=None):
        # If this takes longer than 3s, MiddlewareTimeout is raised
        result = await call_external_api(prompt)
        return prompt

base_agent = Agent("openai:gpt-4o")
agent = MiddlewareAgent(agent=base_agent, middleware=[ExternalAPICheck()])

try:
    result = await agent.run("Check this input")
except MiddlewareTimeout as e:
    print(f"Middleware '{e.middleware_name}' timed out in {e.hook_name} after {e.timeout}s")
```

---

## Architecture

```
 pydantic-ai Agent              pydantic-ai-middleware
┌──────────────────┐    ┌─────────────────────────────────────────┐
│                  │    │                                         │
│  Agent(model,    │    │  MiddlewareAgent(agent, middleware)     │
│    tools,        │◄───│                                         │
│    instructions) │    │  middleware = [                         │
│                  │    │    MiddlewareChain([MW1, MW2])          │
└──────────────────┘    │    ParallelMiddleware([MW3, MW4])       │
                        │    ConditionalMiddleware(cond, MW5)     │
                        │    AsyncGuardrailMiddleware(MW6)        │
                        │  ]                                      │
                        │                                         │
                        │  + MiddlewareContext (data sharing)     │
                        │  + PermissionHandler (tool auth)        │
                        │  + PipelineSpec (config loading)        │
                        └─────────────────────────────────────────┘
```

---

## Use Cases

| What You Want to Build | Key Components |
|------------------------|----------------|
| **Input Validation** | before_run + InputBlocked |
| **PII Redaction** | before_run + after_run |
| **Rate Limiting** | before_run + context + timeout |
| **Tool Authorization** | before_tool_call + ToolPermissionResult |
| **Scoped Tool Guards** | before_tool_call + tool_names |
| **Tool Error Recovery** | on_tool_error + tool_names |
| **Audit Logging** | All hooks + context |
| **Content Moderation** | Parallel + AsyncGuardrail |
| **A/B Testing** | ConditionalMiddleware |
| **Config-Driven Pipelines** | PipelineSpec + Config Loading |

---

## Part of the Ecosystem

| Package | Description |
|---------|-------------|
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation: Agent framework by Pydantic |
| [pydantic-deep](https://github.com/vstorm-co/pydantic-deep) | Full agent framework with planning, subagents, skills |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage and sandbox backends |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning toolset for agents |
| [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) | Multi-agent orchestration |
| [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) | Context management processors |

---

## Contributing

```bash
git clone https://github.com/vstorm-co/pydantic-ai-middleware.git
cd pydantic-ai-middleware
make install
make test  # 100% coverage required
make all   # lint + typecheck + test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Star History

<p align="center">
  <a href="https://www.star-history.com/#vstorm-co/pydantic-ai-middleware&type=date">
    <img src="https://api.star-history.com/svg?repos=vstorm-co/pydantic-ai-middleware&type=date" alt="Star History" width="600">
  </a>
</p>

---

## License

MIT — see [LICENSE](LICENSE)

<p align="center">
  <sub>Built with care by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>

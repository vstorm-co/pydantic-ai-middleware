# pydantic-ai-middleware

> **Looking for a complete agent framework?** Check out [pydantic-deep](https://github.com/vstorm-co/pydantic-deep) - a full-featured deep agent framework with planning, subagents, and skills system built on pydantic-ai.

> **Need task planning tools?** Check out [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) - todo/task planning toolset that works with any pydantic-ai agent.

> **Need file storage or Docker sandbox?** Check out [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) - file storage and sandbox backends that work with any pydantic-ai agent.

[![PyPI version](https://badge.fury.io/py/pydantic-ai-middleware.svg)](https://badge.fury.io/py/pydantic-ai-middleware)
[![Python Versions](https://img.shields.io/pypi/pyversions/pydantic-ai-middleware.svg)](https://pypi.org/project/pydantic-ai-middleware/)
[![CI](https://github.com/vstorm-co/pydantic-ai-middleware/actions/workflows/ci.yml/badge.svg)](https://github.com/vstorm-co/pydantic-ai-middleware/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/vstorm-co/pydantic-ai-middleware)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple middleware library for [Pydantic-AI](https://github.com/pydantic/pydantic-ai) - clean before/after hooks without imposed guardrails structure.

## Features

- **Clean Middleware API** - Simple before/after hooks at every lifecycle stage
- **No Imposed Structure** - You decide what to do (logging, guardrails, metrics, transformations)
- **Full Control** - Modify prompts, outputs, tool calls, and handle errors
- **Decorator Support** - Simple decorators for quick middleware creation
- **Parallel Execution** - Run multiple middleware concurrently with early cancellation
- **Async Guardrails** - Run guardrails concurrently with LLM calls
- **Type Safe** - Full typing support with generics for dependencies

## Installation

```bash
pip install pydantic-ai-middleware
```

Or with uv:

```bash
uv add pydantic-ai-middleware
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware, InputBlocked

class SecurityMiddleware(AgentMiddleware[None]):
    """Block dangerous inputs."""

    async def before_run(self, prompt, deps):
        if "dangerous" in prompt.lower():
            raise InputBlocked("Dangerous content detected")
        return prompt

class LoggingMiddleware(AgentMiddleware[None]):
    """Log agent activity."""

    async def before_run(self, prompt, deps):
        print(f"Starting: {prompt[:50]}...")
        return prompt

    async def after_run(self, prompt, output, deps):
        print(f"Finished: {output}")
        return output

# Create base agent
base_agent = Agent('openai:gpt-4o')

# Wrap with middleware
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[
        LoggingMiddleware(),
        SecurityMiddleware(),
    ],
)

# Use normally
result = await agent.run("Hello, how are you?")
```

## Middleware Hooks

| Hook | When Called | Can Modify |
|------|-------------|------------|
| `before_run` | Before agent starts | Prompt |
| `after_run` | After agent finishes | Output |
| `before_model_request` | Before each model call | Messages |
| `before_tool_call` | Before tool execution | Tool arguments |
| `after_tool_call` | After tool execution | Tool result |
| `on_error` | When error occurs | Exception |

## Parallel Execution

Run multiple middleware concurrently with `ParallelMiddleware`:

```python
from pydantic_ai_middleware import ParallelMiddleware, AggregationStrategy

# Run 3 validators in parallel instead of sequentially
parallel_validators = ParallelMiddleware(
    middleware=[
        ProfanityFilter(),      # 0.3s
        PIIDetector(),          # 0.5s  
        InjectionGuard(),       # 0.3s
    ],
    strategy=AggregationStrategy.ALL_MUST_PASS,
)

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[parallel_validators],
)

# Sequential: 0.3 + 0.5 + 0.3 = 1.1s
# Parallel:   max(0.3, 0.5, 0.3) = 0.5s with early cancellation on failure
```

### Aggregation Strategies

| Strategy | Behavior |
|----------|----------|
| `ALL_MUST_PASS` | All must succeed; cancels remaining on first failure |
| `FIRST_SUCCESS` | Returns first success; cancels remaining tasks |
| `RACE` | Returns first completion (success or failure) |
| `COLLECT_ALL` | Waits for all results |

## Async Guardrails

Run guardrails concurrently with LLM calls using `AsyncGuardrailMiddleware`:

```python
from pydantic_ai_middleware import AsyncGuardrailMiddleware, GuardrailTiming

# Run safety check in parallel with LLM - cancel LLM if guardrail fails
guardrail = AsyncGuardrailMiddleware(
    guardrail=SafetyChecker(),
    timing=GuardrailTiming.CONCURRENT,
    cancel_on_failure=True,
)

# If guardrail fails at 0.5s while LLM is still running,
# the LLM call is cancelled immediately, saving time and costs
```

### Timing Modes

| Mode | Behavior |
|------|----------|
| `BLOCKING` | Guardrail completes before LLM starts |
| `CONCURRENT` | Guardrail runs alongside LLM; can cancel on failure |
| `ASYNC_POST` | Guardrail runs in background after response |

## Decorator Syntax

For simple cases, use decorators:

```python
from pydantic_ai_middleware import before_run, after_run, before_tool_call, ToolBlocked

@before_run
async def log_input(prompt, deps):
    print(f"Input: {prompt}")
    return prompt

@after_run
async def log_output(prompt, output, deps):
    print(f"Output: {output}")
    return output

@before_tool_call
async def validate_tools(tool_name, tool_args, deps):
    if tool_name == "dangerous_tool":
        raise ToolBlocked(tool_name, "Not allowed")
    return tool_args
```

## Middleware Execution Order

Middleware executes in order for `before_*` hooks and reverse order for `after_*` hooks:

```python
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[
        RateLimitMiddleware(),   # 1st before, last after
        LoggingMiddleware(),     # 2nd before, 2nd-to-last after
        SecurityMiddleware(),    # 3rd before, 3rd-to-last after
    ],
)

# before_run order: RateLimit -> Logging -> Security -> [Agent]
# after_run order:  [Agent] -> Security -> Logging -> RateLimit
```

## Example Middleware

### Rate Limiting

```python
import time
from pydantic_ai_middleware import AgentMiddleware

class RateLimitMiddleware(AgentMiddleware[None]):
    def __init__(self, max_calls: int = 10, window: int = 60):
        self.max_calls = max_calls
        self.window = window
        self._calls: list[float] = []

    async def before_run(self, prompt, deps):
        now = time.time()
        self._calls = [t for t in self._calls if now - t < self.window]

        if len(self._calls) >= self.max_calls:
            raise Exception("Rate limit exceeded")

        self._calls.append(now)
        return prompt
```

### Tool Authorization

```python
from pydantic_ai_middleware import AgentMiddleware, ToolBlocked

class ToolAuthMiddleware(AgentMiddleware[MyDeps]):
    dangerous_tools = {"delete_file", "execute_code", "send_email"}

    async def before_tool_call(self, tool_name, tool_args, deps):
        if tool_name in self.dangerous_tools:
            if not deps.user.is_admin:
                raise ToolBlocked(tool_name, "Requires admin privileges")
        return tool_args
```

### Error Handling

```python
from pydantic_ai_middleware import AgentMiddleware

class ErrorHandlerMiddleware(AgentMiddleware[MyDeps]):
    async def on_error(self, error, deps):
        # Log error
        await error_tracker.report(error, user_id=deps.user_id)

        # Convert to user-friendly message
        if isinstance(error, RateLimitError):
            return UserFacingError("Service busy, try later")

        return None  # Re-raise original
```

## Building Guardrails

This library provides flexible building blocks for implementing guardrails without imposing a rigid structure. You decide what guardrails you need and how they behave.

### Input Validation Guardrail

```python
from pydantic_ai_middleware import AgentMiddleware, InputBlocked

class InputValidationGuardrail(AgentMiddleware[MyDeps]):
    """Validate and sanitize user input before processing."""

    async def before_run(self, prompt, deps):
        # Check for profanity
        if has_profanity(prompt):
            raise InputBlocked("Inappropriate content detected")

        # Redact PII (emails, phone numbers, SSNs)
        prompt = redact_pii(prompt)

        # Length validation
        if len(prompt) > 10000:
            raise InputBlocked("Input too long")

        return prompt
```

### Content Moderation Guardrail

Use a separate AI model to moderate content before processing:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_middleware import AgentMiddleware, InputBlocked

class ModerationResult(BaseModel):
    is_safe: bool
    reason: str | None = None

class ContentModerationGuardrail(AgentMiddleware[None]):
    """Use AI to moderate content before processing."""

    def __init__(self):
        self.moderator = Agent(
            'openai:gpt-4o-mini',
            output_type=ModerationResult,
            system_prompt="Analyze if the content is safe. Return is_safe=False for harmful content.",
        )

    async def before_run(self, prompt, deps):
        result = await self.moderator.run(str(prompt))
        if not result.output.is_safe:
            raise InputBlocked(result.output.reason or "Content not allowed")
        return prompt
```

### PII Redaction Guardrail

```python
import re
from pydantic_ai_middleware import AgentMiddleware

class PIIRedactionGuardrail(AgentMiddleware[None]):
    """Redact personally identifiable information from prompts and outputs."""

    patterns = {
        'email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    }

    def redact(self, text: str) -> str:
        for name, pattern in self.patterns.items():
            text = re.sub(pattern, f'[{name.upper()}_REDACTED]', text)
        return text

    async def before_run(self, prompt, deps):
        if isinstance(prompt, str):
            return self.redact(prompt)
        return prompt

    async def after_run(self, prompt, output, deps):
        if isinstance(output, str):
            return self.redact(output)
        return output
```

### Audit Logging Guardrail

```python
from datetime import datetime
from pydantic_ai_middleware import AgentMiddleware

class AuditGuardrail(AgentMiddleware[MyDeps]):
    """Log all agent activity for compliance and debugging."""

    async def before_run(self, prompt, deps):
        await audit_log.record(
            user_id=deps.user_id,
            action="agent:start",
            input_summary=str(prompt)[:100],
            timestamp=datetime.now(),
        )
        return prompt

    async def before_tool_call(self, tool_name, tool_args, deps):
        await audit_log.record(
            user_id=deps.user_id,
            action=f"tool:{tool_name}",
            params=tool_args,
            timestamp=datetime.now(),
        )
        return tool_args

    async def after_run(self, prompt, output, deps):
        await audit_log.record(
            user_id=deps.user_id,
            action="agent:complete",
            output_summary=str(output)[:100],
            timestamp=datetime.now(),
        )
        return output
```

## Middleware vs Traditional Guardrails

| Aspect | Middleware (this library) | Traditional Guardrails |
|--------|---------------------------|------------------------|
| Complexity | Low | High |
| Structure | No imposed structure | Fixed result types, actions |
| Flexibility | Maximum | Constrained by design |
| Learning curve | Flat | Steeper |
| Built-in guardrails | None (you build what you need) | Pre-built (PII, moderation) |
| Parallel execution | Built-in with early cancellation | Often built-in |
| Type safety | Full generics support | Varies |

### When to Use This Library

- You want **simple hooks** without complex abstractions
- You need **full control** over behavior
- You prefer **building custom guardrails** over using pre-built ones
- Your use case is **logging, metrics, rate limiting, or basic validation**
- You want **minimal dependencies** and learning curve

### When to Consider Full Guardrails Libraries

- You need **pre-built guardrails** (PII detection, content moderation)
- You want **parallel execution** of multiple guardrails
- You need **human-in-the-loop** approval workflows
- You prefer a **standardized API** with built-in retry logic

## Development

```bash
# Install dependencies
make install

# Run tests
make test

# Run all checks
make all
```

## Related Projects

- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** - The foundation: Agent framework by Pydantic
- **[pydantic-deep](https://github.com/vstorm-co/pydantic-deep)** - Full agent framework with planning, subagents, and skills
- **[pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo)** - Todo/task planning toolset for agents
- **[pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend)** - File storage and sandbox backends

## License

[MIT](LICENSE)

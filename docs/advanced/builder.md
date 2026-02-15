# Builder / Factory API

The builder module provides the registry, compiler, and node handler system that
powers config-driven middleware pipelines. Use it when you need programmatic control
over how middleware is constructed from configuration data.

## MiddlewareRegistry

`MiddlewareRegistry` is the user-facing extension mechanism. Register middleware
factories and predicate functions, then pass the registry to the compiler.

```python
from pydantic_ai_middleware import AgentMiddleware
from pydantic_ai_middleware.builder import MiddlewareRegistry


class LoggingMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        print(f"Input: {prompt}")
        return prompt


class RateLimiter(AgentMiddleware[None]):
    def __init__(self, max_calls: int = 10):
        self.max_calls = max_calls

    async def before_run(self, prompt, deps, ctx):
        return prompt


reg: MiddlewareRegistry[None] = MiddlewareRegistry()

# Register by name
reg.register_middleware("logging", LoggingMiddleware)
reg.register_middleware("rate_limit", RateLimiter)
```

### Decorator registration

Use the `@middleware_factory` decorator for a cleaner syntax:

```python
from pydantic_ai_middleware.builder import MiddlewareRegistry

reg: MiddlewareRegistry[None] = MiddlewareRegistry()


@reg.middleware_factory("logging")
def logging_factory(**config):
    return LoggingMiddleware()


@reg.middleware_factory("rate_limit")
def rate_limit_factory(max_calls: int = 10, **config):
    return RateLimiter(max_calls=max_calls)
```

### Predicate registration

Predicates are used by `when` nodes for conditional branching:

```python
@reg.predicate("is_admin")
def is_admin(ctx):
    return ctx is not None and ctx.config.get("role") == "admin"

# Or register directly
reg.register_predicate("always_true", lambda ctx: True)
```

Predicates can also be factories that accept configuration:

```python
@reg.predicate("has_role")
def has_role(role: str):
    def check(ctx):
        return ctx is not None and ctx.config.get("role") == role
    return check
```

When referenced from config, factory predicates receive their parameters via a
`config` mapping:

```json
{
  "when": {
    "predicate": {"name": "has_role", "config": {"role": "admin"}},
    "then": [{"type": "audit"}]
  }
}
```

## MiddlewarePipelineCompiler

The compiler turns config specs (dicts/lists) into middleware instances. It uses the
registry to look up factories and predicates.

```python
from pydantic_ai_middleware.builder import (
    MiddlewarePipelineCompiler,
    MiddlewareRegistry,
)

reg: MiddlewareRegistry[None] = MiddlewareRegistry()
reg.register_middleware("logging", LoggingMiddleware)
reg.register_middleware("rate_limit", RateLimiter)

compiler = MiddlewarePipelineCompiler(registry=reg)

# Compile a single node
middleware = compiler.compile({"type": "logging"})

# Compile a list of nodes
middleware_list = compiler.compile_list([
    {"type": "logging"},
    {"type": "rate_limit", "config": {"max_calls": 50}},
])
```

### compile vs compile_list

| Method | Input | Output |
|---|---|---|
| `compile()` | A single mapping or a sequence of mappings | `AgentMiddleware` or `list[AgentMiddleware]` |
| `compile_list()` | A mapping, sequence, or `AgentMiddleware` instance | Always `list[AgentMiddleware]` |

Use `compile_list()` when you need a flat list regardless of input shape.

## Node types

The compiler recognises four built-in node types. Each is a top-level key in a
config mapping.

### type

Instantiate a registered middleware by name. Pass constructor arguments via `config`:

```json
{"type": "rate_limit", "config": {"max_calls": 100}}
```

```python
compiler.compile({"type": "rate_limit", "config": {"max_calls": 100}})
```

### chain

Sequential list of middleware nodes. The result is a flat list of middleware
instances that run in order:

```json
{
  "chain": [
    {"type": "logging"},
    {"type": "rate_limit", "config": {"max_calls": 50}}
  ]
}
```

```python
middleware_list = compiler.compile_list({"chain": [
    {"type": "logging"},
    {"type": "rate_limit", "config": {"max_calls": 50}},
]})
```

### parallel

Execute multiple middleware concurrently and aggregate results using a strategy:

```json
{
  "parallel": {
    "middleware": [
      {"type": "pii_guard"},
      {"type": "profanity_filter"}
    ],
    "strategy": "all_must_pass",
    "timeout": 5.0,
    "name": "SecurityChecks"
  }
}
```

Available aggregation strategies:

| Strategy | Behaviour |
|---|---|
| `all_must_pass` | All middleware must succeed (default). |
| `any_must_pass` | At least one must succeed. |
| `majority_must_pass` | More than half must succeed. |
| `first_result` | Use the first result that completes. |

### when

Conditional branching based on a predicate. The predicate receives the current
`ScopedContext` and returns a boolean:

```json
{
  "when": {
    "predicate": "is_admin",
    "then": [{"type": "admin_audit"}],
    "else": [{"type": "basic_audit"}]
  }
}
```

The `else` branch is optional. Predicates can be:

- A string name referencing a registered predicate.
- A mapping with `name` and `config` keys for parameterised predicates.
- A boolean literal (`true` / `false`).
- A callable (when building config in Python).

## Custom node handlers

For advanced use cases you can register custom node handlers on the compiler:

```python
from pydantic_ai_middleware.builder import MiddlewarePipelineCompiler

def _handle_retry(compiler, spec):
    """Custom node: wrap middleware with retry logic."""
    retry_spec = spec["retry"]
    inner = compiler.compile_list(retry_spec["middleware"])
    max_retries = retry_spec.get("max_retries", 3)
    # Return a custom retry wrapper
    return RetryMiddleware(inner, max_retries=max_retries)

compiler.register_node_handler("retry", _handle_retry)

# Now this config is valid:
# {"retry": {"middleware": [{"type": "flaky_api"}], "max_retries": 5}}
```

## Backwards-compatible helpers

Two module-level functions provide a simpler API that creates a registry and compiler
internally:

```python
from pydantic_ai_middleware.builder import build_middleware, build_middleware_list

registry = {"logging": LoggingMiddleware, "rate_limit": RateLimiter}

# Returns AgentMiddleware or list[AgentMiddleware]
mw = build_middleware({"type": "logging"}, registry=registry)

# Always returns list[AgentMiddleware]
mw_list = build_middleware_list(
    [{"type": "logging"}, {"type": "rate_limit"}],
    registry=registry,
)
```

## Next Steps

- [Config Loading](config-loading.md) - Load pipelines from JSON/YAML files
- [Pipeline Spec](pipeline-spec.md) - Fluent API for building pipeline specs
- [Conditional Routing](conditional-middleware.md) - Runtime branching

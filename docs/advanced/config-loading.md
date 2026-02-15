# Config Loading

Load middleware pipelines from JSON or YAML files. This makes it easy to share the
same pipeline across environments without duplicating code.

## Supported nodes

- `{"type": "...", "config": {...}}`
- `{"chain": [ ... ]}`
- `{"parallel": {"middleware": [ ... ], "strategy": "...", "timeout": 1.5}}`
- `{"when": {"predicate": "...", "then": [ ... ], "else": [ ... ]}}`

## Example: JSON pipeline

```json
[
  {"type": "logging"},
  {
    "parallel": {
      "middleware": [
        {"type": "pii_guard"},
        {"type": "profanity_filter"}
      ],
      "strategy": "all_must_pass"
    }
  },
  {
    "when": {
      "predicate": "is_admin",
      "then": [{"type": "audit"}],
      "else": [{"type": "basic_audit"}]
    }
  }
]
```

## Example: YAML pipeline

```yaml
- type: logging
- parallel:
    middleware:
      - type: pii_guard
      - type: profanity_filter
    strategy: all_must_pass
```

## Loading from a file

Use `load_middleware_config_path()` to read a JSON or YAML file and compile it into
middleware instances. The format is detected automatically from the file extension
(`.json`, `.yaml`, `.yml`).

```python
from pydantic_ai_middleware import load_middleware_config_path

registry = {
    "logging": LoggingMiddleware,
    "pii_guard": PIIGuardMiddleware,
    "profanity_filter": ProfanityFilter,
}

middleware = load_middleware_config_path("pipeline.yaml", registry=registry)
```

You can also force a specific format:

```python
middleware = load_middleware_config_path(
    "pipeline.conf",
    registry=registry,
    format="yaml",
)
```

## Loading from text

Use `load_middleware_config_text()` when you already have the config as a string
(for example, read from a database or environment variable):

```python
from pydantic_ai_middleware import load_middleware_config_text

config_text = '[{"type": "logging"}]'
middleware = load_middleware_config_text(config_text, registry=registry)
```

The format is auto-detected by examining the first non-whitespace character: `{` or
`[` means JSON, anything else is treated as YAML. Override with the `format` parameter
if needed.

## Format detection

| Signal | Detected format |
|---|---|
| `format="json"` parameter | JSON |
| `format="yaml"` or `format="yml"` parameter | YAML |
| File extension `.json` | JSON |
| File extension `.yaml` or `.yml` | YAML |
| Text starts with `{` or `[` | JSON |
| Anything else | YAML |

!!! note
    YAML support requires the `PyYAML` package. If it is not installed a
    `MiddlewareConfigError` is raised when loading YAML configs.

## Registering middleware factories

Use `register_middleware()` to add factories to a plain dict registry. It works as a
direct call or as a decorator:

```python
from pydantic_ai_middleware import register_middleware

registry: dict[str, Any] = {}

# Direct registration
register_middleware(registry, LoggingMiddleware, name="logging")

# Decorator registration (name defaults to function name)
@register_middleware(registry)
def rate_limit(max_calls: int = 10, **kwargs):
    return RateLimiter(max_calls=max_calls)
```

## Registering predicates

Use `register_predicate()` to register predicate functions used by `when` nodes:

```python
from pydantic_ai_middleware import register_predicate

predicates: dict[str, Any] = {}

@register_predicate(predicates)
def is_admin(ctx):
    return ctx is not None and ctx.config.get("role") == "admin"

middleware = load_middleware_config_path(
    "pipeline.yaml",
    registry=registry,
    predicates=predicates,
)
```

## Registry and compiler

For more control, use `MiddlewareRegistry` and `MiddlewarePipelineCompiler` directly:

```python
from pydantic_ai_middleware import MiddlewarePipelineCompiler, MiddlewareRegistry

reg = MiddlewareRegistry(middleware=registry)
compiler = MiddlewarePipelineCompiler(registry=reg)
middleware = compiler.compile(config_data)
middleware_list = compiler.compile_list(config_data)
```

See the [Builder / Factory API](builder.md) page for full details on the registry and
compiler.

## Predicates for conditional nodes

```python
from pydantic_ai_middleware import MiddlewareRegistry

reg = MiddlewareRegistry(middleware=registry)

@reg.predicate("is_admin")
def is_admin(ctx):
    return ctx is not None and ctx.config.get("role") == "admin"

middleware = load_middleware_config_text(
    config_text,
    registry=reg.middleware,
    predicates=reg.predicates,
)
```

## Next Steps

- [Builder / Factory API](builder.md) - Registry, compiler, and custom node handlers
- [Pipeline Spec](pipeline-spec.md) - Build and export pipeline specs
- [Conditional Routing](conditional-middleware.md) - Runtime branching

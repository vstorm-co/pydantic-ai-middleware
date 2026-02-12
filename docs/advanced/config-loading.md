# Config Loading

You can load middleware pipelines from JSON or YAML. This makes it easy to share
the same pipeline across environments without duplicating code.

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

## Load from text or file

```python
from pydantic_ai_middleware import (
    load_middleware_config_path,
    load_middleware_config_text,
)

registry = {
    "logging": LoggingMiddleware,
    "pii_guard": PIIGuardMiddleware,
    "profanity_filter": ProfanityFilter,
}

middleware = load_middleware_config_text(config_text, registry=registry)
middleware = load_middleware_config_path("pipeline.yaml", registry=registry)
```

## Registry and compiler

```python
from pydantic_ai_middleware import MiddlewarePipelineCompiler, MiddlewareRegistry

reg = MiddlewareRegistry(middleware=registry)
compiler = MiddlewarePipelineCompiler(registry=reg)
middleware = compiler.compile(config_data)
middleware_list = compiler.compile_list(config_data)
```

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

- [Pipeline Spec](pipeline-spec.md) - Build and export pipeline specs
- [Conditional Routing](conditional-middleware.md) - Runtime branching

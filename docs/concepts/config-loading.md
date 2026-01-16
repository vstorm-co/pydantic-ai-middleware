# Config Loading

You can load and save middleware pipelines as JSON or YAML. This makes it easy to share
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

## Predicates for conditional nodes

```python
from pydantic_ai_middleware import register_predicate

predicates: dict[str, object] = {}

@register_predicate(predicates)
def is_admin(ctx):
    return ctx is not None and ctx.config.get("role") == "admin"

middleware = load_middleware_config_text(
    config_text,
    registry=registry,
    predicates=predicates,
)
```

## Save with stable output

```python
from pydantic_ai_middleware import save_middleware_config_path

save_middleware_config_path(config_data, "pipeline.json")
save_middleware_config_path(config_data, "pipeline.yaml")
```

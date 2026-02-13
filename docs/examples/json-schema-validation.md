# JSON Schema Validation

This example demonstrates middleware that validates tool arguments and agent outputs against JSON schemas, ensuring structured data conforms to expected formats before processing.

## Overview

The `JsonSchemaMiddleware` registers schemas per tool name. When a tool is called, its arguments are validated against the registered schema. Invalid calls are blocked with `ToolBlocked`.

## Two Modes

| Mode | Behavior |
|------|----------|
| `strict=True` (default) | Block on validation failure with `ToolBlocked` |
| `strict=False` | Log warning but allow the call through |

## Basic Usage

```python
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai_middleware import AgentMiddleware, ToolBlocked, ScopedContext


@dataclass
class SchemaValidationConfig:
    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    strict: bool = True


class JsonSchemaMiddleware(AgentMiddleware[None]):
    def __init__(
        self, config: SchemaValidationConfig | None = None
    ) -> None:
        self.config = config or SchemaValidationConfig()

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        schema = self.config.tool_schemas.get(tool_name)
        if schema is None:
            return tool_args  # no schema -> pass through

        errors = validate_value(tool_args, schema)

        if errors and self.config.strict:
            details = "; ".join(errors)
            raise ToolBlocked(
                tool_name,
                f"Schema validation failed: {details}",
            )

        return tool_args
```

## Defining Schemas

```python
SEND_EMAIL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["to", "subject", "body"],
    "properties": {
        "to": {
            "type": "string",
            "pattern": r"^[\w.-]+@[\w.-]+\.\w+$",
        },
        "subject": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
        },
        "body": {
            "type": "string",
            "minLength": 1,
        },
    },
}

CREATE_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["username", "age", "role"],
    "properties": {
        "username": {
            "type": "string",
            "minLength": 3,
            "maxLength": 32,
        },
        "age": {
            "type": "integer",
            "minimum": 13,
            "maximum": 150,
        },
        "role": {
            "type": "string",
            "enum": ["user", "admin", "moderator"],
        },
    },
}
```

## Usage with Agent

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent

config = SchemaValidationConfig(
    tool_schemas={
        "send_email": SEND_EMAIL_SCHEMA,
        "create_user": CREATE_USER_SCHEMA,
    }
)

agent = MiddlewareAgent(
    agent=Agent('openai:gpt-4o', output_type=str),
    middleware=[JsonSchemaMiddleware(config)],
)
```

## Warn Mode

Use `strict=False` to log validation errors without blocking:

```python
config = SchemaValidationConfig(
    tool_schemas={"create_user": CREATE_USER_SCHEMA},
    strict=False,  # Log but don't block
)
```

## Runnable Example

See `examples/json_schema_validation.py` for a complete runnable demo:

```bash
uv run python examples/json_schema_validation.py
```

"""JSON schema enforcement middleware example.

Demonstrates how to build middleware that validates tool arguments
and agent outputs against JSON schemas, ensuring structured data
conforms to expected formats before processing.

Run with: uv run python examples/json_schema_validation.py
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    MiddlewareAgent,
    ScopedContext,
    ToolBlocked,
)

# ---------------------------------------------------------------------------
# Lightweight JSON Schema validator (no external dependency)
# ---------------------------------------------------------------------------


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _check_type(value: Any, schema: dict[str, Any], path: str) -> str | None:
    """Return an error message if the value doesn't match the expected type."""
    expected = schema.get("type")
    if expected and expected in _TYPE_MAP and not isinstance(value, _TYPE_MAP[expected]):
        return f"{path}: expected {expected}, got {type(value).__name__}"
    return None


def _check_constraints(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    """Check enum, string, and numeric constraints."""
    import re

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string too short (min {schema['minLength']})")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string too long (max {schema['maxLength']})")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    return errors


def validate_value(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate a value against a JSON-schema-like dict.

    Supports: type, required, properties, items, minimum, maximum,
    minLength, maxLength, enum, pattern.
    """
    type_error = _check_type(value, schema, path)
    if type_error:
        return [type_error]

    errors = _check_constraints(value, schema, path)

    if isinstance(value, dict):
        required = set(schema.get("required", []))
        for req_key in required:
            if req_key not in value:
                errors.append(f"{path}.{req_key}: required field missing")
        props = schema.get("properties", {})
        for key, val in value.items():
            if key in props:
                errors.extend(validate_value(val, props[key], f"{path}.{key}"))

    if isinstance(value, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                errors.extend(validate_value(item, items_schema, f"{path}[{i}]"))

    return errors


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@dataclass
class SchemaValidationConfig:
    """Configuration for JSON schema validation middleware."""

    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    strict: bool = True  # block on violation vs warn


class JsonSchemaMiddleware(AgentMiddleware[None]):
    """Middleware that validates tool arguments against JSON schemas.

    Register schemas per tool name. When a tool is called, its arguments
    are validated against the registered schema. Invalid calls are blocked
    with :class:`ToolBlocked`.
    """

    def __init__(self, config: SchemaValidationConfig | None = None) -> None:
        self.config = config or SchemaValidationConfig()

    async def before_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> dict[str, Any]:
        schema = self.config.tool_schemas.get(tool_name)
        if schema is None:
            return tool_args  # no schema registered -> pass through

        errors = validate_value(tool_args, schema)

        if ctx:
            ctx.set(f"{tool_name}_validation_errors", errors)

        if errors:
            details = "; ".join(errors)
            print(f"    [Schema] {tool_name}: {details}")
            if self.config.strict:
                raise ToolBlocked(tool_name, f"Schema validation failed: {details}")

        return tool_args


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


# Define schemas for tools
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

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
        },
    },
}


async def demo_tool_validation() -> None:
    """Demonstrate tool argument schema validation."""
    print("=" * 60)
    print("TOOL ARGUMENT SCHEMA VALIDATION")
    print("=" * 60)

    config = SchemaValidationConfig(
        tool_schemas={
            "send_email": SEND_EMAIL_SCHEMA,
            "create_user": CREATE_USER_SCHEMA,
            "search": SEARCH_SCHEMA,
        }
    )
    middleware = JsonSchemaMiddleware(config)

    test_cases: list[tuple[str, dict[str, Any], bool]] = [
        # Valid calls
        (
            "send_email",
            {"to": "user@example.com", "subject": "Hello", "body": "Hi there!"},
            False,
        ),
        (
            "create_user",
            {"username": "john_doe", "age": 25, "role": "user"},
            False,
        ),
        (
            "search",
            {"query": "python tutorials", "limit": 10},
            False,
        ),
        # Invalid calls
        (
            "send_email",
            {"to": "not-an-email", "subject": "Hello", "body": "Hi"},
            True,
        ),
        (
            "send_email",
            {"to": "user@example.com", "body": "Missing subject"},
            True,
        ),
        (
            "create_user",
            {"username": "ab", "age": 5, "role": "superadmin"},
            True,
        ),
        (
            "search",
            {"query": "", "limit": 999},
            True,
        ),
        # Unknown tool (no schema -> pass through)
        (
            "unknown_tool",
            {"anything": "goes"},
            False,
        ),
    ]

    for tool_name, tool_args, should_fail in test_cases:
        print(f"\n  Tool: {tool_name}({json.dumps(tool_args, indent=None)})")
        try:
            await middleware.before_tool_call(tool_name, tool_args, None)
            status = "PASSED" if not should_fail else "UNEXPECTED PASS"
            print(f"  -> [{status}] args accepted")
        except ToolBlocked as e:
            status = "BLOCKED" if should_fail else "UNEXPECTED BLOCK"
            print(f"  -> [{status}] {e.reason}")


async def demo_warn_mode() -> None:
    """Demonstrate non-strict (warn) mode."""
    print("\n\n" + "=" * 60)
    print("WARN MODE (strict=False, logs but doesn't block)")
    print("=" * 60)

    config = SchemaValidationConfig(
        tool_schemas={"create_user": CREATE_USER_SCHEMA},
        strict=False,
    )
    middleware = JsonSchemaMiddleware(config)

    # Invalid arguments - should warn but not block
    bad_args = {"username": "x", "age": 5, "role": "superadmin"}
    print(f"\n  Tool: create_user({json.dumps(bad_args)})")
    result = await middleware.before_tool_call("create_user", bad_args, None)
    print(f"  -> PASSED (warn mode) - args returned: {result}")


async def demo_with_agent() -> None:
    """Demonstrate schema validation integrated with a MiddlewareAgent."""
    print("\n\n" + "=" * 60)
    print("INTEGRATED WITH MiddlewareAgent")
    print("=" * 60)

    model = TestModel()
    model.custom_output_text = "Email sent successfully!"

    config = SchemaValidationConfig(
        tool_schemas={"send_email": SEND_EMAIL_SCHEMA},
    )

    agent = MiddlewareAgent(
        agent=Agent(model, output_type=str),
        middleware=[JsonSchemaMiddleware(config)],
    )

    result = await agent.run("Send an email to the team")
    print(f"\n  Agent output: {result.output}")
    print("  (Schema validation is active for any tool calls the agent makes)")


async def main() -> None:
    await demo_tool_validation()
    await demo_warn_mode()
    await demo_with_agent()
    print()


if __name__ == "__main__":
    asyncio.run(main())

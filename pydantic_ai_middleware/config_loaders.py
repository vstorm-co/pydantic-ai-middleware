"""Load and save middleware pipelines from JSON/YAML configs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import AgentMiddleware, DepsT
from .builder import MiddlewareFactory, Predicate, PredicateFactory, build_middleware_list
from .exceptions import MiddlewareConfigError


def load_middleware_config_text(
    text: str,
    *,
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
    format: str | None = None,
) -> list[AgentMiddleware[DepsT]]:
    """Load middleware pipeline from raw JSON/YAML text."""
    fmt = _detect_format(text=text, format=format)
    raw = _load_text(text, fmt)
    return build_middleware_list(raw, registry=registry, predicates=predicates)


def load_middleware_config_path(
    path: str | Path,
    *,
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
    format: str | None = None,
) -> list[AgentMiddleware[DepsT]]:
    """Load middleware pipeline from a file path."""
    config_path = Path(path)
    text = config_path.read_text()
    fmt = _detect_format(text=text, format=format, path=config_path)
    raw = _load_text(text, fmt)
    return build_middleware_list(raw, registry=registry, predicates=predicates)


def dump_middleware_config(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    format: str = "json",
) -> str:
    """Dump middleware config to JSON/YAML with stable output."""
    fmt = _normalize_format(format)
    normalized = _normalize_config(config)
    if fmt == "json":
        return json.dumps(normalized, indent=2, sort_keys=True)
    return _dump_yaml(normalized)


def save_middleware_config_path(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    format: str | None = None,
) -> None:
    """Save middleware config to a JSON/YAML file."""
    config_path = Path(path)
    fmt = _detect_format(format=format, path=config_path)
    text = dump_middleware_config(config, format=fmt)
    config_path.write_text(text)


def register_middleware(
    registry: dict[str, MiddlewareFactory[DepsT]],
    factory: MiddlewareFactory[DepsT] | None = None,
    *,
    name: str | None = None,
) -> Callable[[MiddlewareFactory[DepsT]], MiddlewareFactory[DepsT]] | MiddlewareFactory[DepsT]:
    """Register a middleware factory in a registry (decorator-friendly)."""

    def _register(item: MiddlewareFactory[DepsT]) -> MiddlewareFactory[DepsT]:
        key = name or item.__name__
        if key in registry:
            raise MiddlewareConfigError(f"Middleware '{key}' is already registered")
        registry[key] = item
        return item

    if factory is not None:
        return _register(factory)
    return _register


def register_predicate(
    registry: dict[str, PredicateFactory | Predicate],
    predicate: PredicateFactory | Predicate | None = None,
    *,
    name: str | None = None,
) -> (
    Callable[[PredicateFactory | Predicate], PredicateFactory | Predicate]
    | PredicateFactory
    | Predicate
):
    """Register a predicate in a registry (decorator-friendly)."""

    def _register(item: PredicateFactory | Predicate) -> PredicateFactory | Predicate:
        key = name or getattr(item, "__name__", None)
        if not key:
            raise MiddlewareConfigError("Predicate name is required for registration")
        if key in registry:
            raise MiddlewareConfigError(f"Predicate '{key}' is already registered")
        registry[key] = item
        return item

    if predicate is not None:
        return _register(predicate)
    return _register


def _detect_format(
    *,
    text: str | None = None,
    format: str | None = None,
    path: Path | None = None,
) -> str:
    if format:
        return _normalize_format(format)

    if path is not None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return "json"
        if suffix in {".yaml", ".yml"}:
            return "yaml"

    if text is not None:
        stripped = text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"
        return "yaml"

    raise MiddlewareConfigError("Unable to determine config format")


def _normalize_format(format: str) -> str:
    fmt = format.lower()
    if fmt == "json":
        return "json"
    if fmt in {"yaml", "yml"}:
        return "yaml"
    raise MiddlewareConfigError(f"Unknown config format: {format!r}")


def _load_text(text: str, fmt: str) -> Any:
    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MiddlewareConfigError(f"Invalid JSON: {exc}") from exc
    return _load_yaml(text)


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise MiddlewareConfigError("YAML support requires PyYAML") from exc
    return yaml.safe_load(text)


def _dump_yaml(data: Any) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise MiddlewareConfigError("YAML support requires PyYAML") from exc
    return yaml.safe_dump(data, sort_keys=True)


def _normalize_config(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]] | Mapping[str, Any]:
    if isinstance(config, Mapping):
        return _normalize_node(config)
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        return [_normalize_node(item) for item in config]
    raise MiddlewareConfigError(
        f"Config must be a mapping or sequence of mappings, got {type(config).__name__}"
    )


def _normalize_node(node: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: dict[str, Any] = dict(node)

    if "chain" in normalized:
        chain = normalized["chain"]
        normalized["chain"] = _normalize_list_node(chain, "chain")

    if "parallel" in normalized:
        parallel = normalized["parallel"]
        if not isinstance(parallel, Mapping):
            raise MiddlewareConfigError(
                f"parallel spec must be a mapping, got {type(parallel).__name__}"
            )
        parallel_dict = dict(parallel)
        if "middleware" in parallel_dict:
            parallel_dict["middleware"] = _normalize_list_node(
                parallel_dict["middleware"], "parallel.middleware"
            )
        normalized["parallel"] = parallel_dict

    if "when" in normalized:
        when_node = normalized["when"]
        if not isinstance(when_node, Mapping):
            raise MiddlewareConfigError(
                f"when spec must be a mapping, got {type(when_node).__name__}"
            )
        when_dict = dict(when_node)
        if "then" in when_dict:
            when_dict["then"] = _normalize_list_node(when_dict["then"], "when.then")
        if "else" in when_dict and when_dict["else"] is not None:
            when_dict["else"] = _normalize_list_node(when_dict["else"], "when.else")
        normalized["when"] = when_dict

    return normalized


def _normalize_list_node(value: Any, label: str) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        raise MiddlewareConfigError(
            f"{label} must be a mapping or sequence, got {type(value).__name__}"
        )

    result: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise MiddlewareConfigError(
                f"{label} items must be mappings, got {type(item).__name__}"
            )
        result.append(_normalize_node(item))
    return result

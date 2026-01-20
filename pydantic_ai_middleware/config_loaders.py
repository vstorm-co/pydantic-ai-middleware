"""Load middleware pipelines from JSON/YAML configs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .base import AgentMiddleware, DepsT
from .builder import MiddlewareFactory, Predicate, PredicateFactory, build_middleware_list
from .exceptions import MiddlewareConfigError


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MiddlewareConfigError("YAML support requires PyYAML") from exc
    return yaml.safe_load(text)


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
    text = config_path.read_text(encoding="utf-8")
    fmt = _detect_format(text=text, format=format, path=config_path)
    raw = _load_text(text, fmt)
    return build_middleware_list(raw, registry=registry, predicates=predicates)


_FactoryT = TypeVar("_FactoryT", bound=Callable[..., Any])
_PredicateT = TypeVar("_PredicateT", bound=Callable[..., Any])


def register_middleware(
    registry: dict[str, Any],
    factory: _FactoryT | None = None,
    *,
    name: str | None = None,
) -> Callable[[_FactoryT], _FactoryT] | _FactoryT:
    """Register a middleware factory in a registry (decorator-friendly)."""

    def _register(item: _FactoryT) -> _FactoryT:
        key = name or item.__name__
        if key in registry:
            raise MiddlewareConfigError(f"Middleware '{key}' is already registered")
        registry[key] = item
        return item

    if factory is not None:
        return _register(factory)
    return _register


def register_predicate(
    registry: dict[str, Any],
    predicate: _PredicateT | None = None,
    *,
    name: str | None = None,
) -> Callable[[_PredicateT], _PredicateT] | _PredicateT:
    """Register a predicate in a registry (decorator-friendly)."""

    def _register(item: _PredicateT) -> _PredicateT:
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

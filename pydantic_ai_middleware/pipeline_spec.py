"""Python-first pipeline spec helpers.

Use this module when you want to define a middleware pipeline *in Python* and:
- compile it into middleware instances using :class:`MiddlewarePipelineCompiler`
- export it as a portable JSON/YAML config spec using `dump()`/`save()`

This intentionally models the same fixed skeleton nodes used by config loading:
`type`, `chain`, `parallel`, `when`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typing_extensions import Self

from .base import AgentMiddleware
from .builder import MiddlewarePipelineCompiler
from .exceptions import MiddlewareConfigError
from .strategies import AggregationStrategy


def type_node(type_name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    node: dict[str, Any] = {"type": type_name}
    if config:
        node["config"] = dict(config)
    return node


def chain_node(nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"chain": [dict(n) for n in nodes]}


def parallel_node(
    nodes: Sequence[Mapping[str, Any]],
    *,
    strategy: str | AggregationStrategy | None = None,
    timeout: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    parallel: dict[str, Any] = {"middleware": [dict(n) for n in nodes]}
    if strategy is not None:
        parallel["strategy"] = (
            strategy.value if isinstance(strategy, AggregationStrategy) else strategy
        )
    if timeout is not None:
        parallel["timeout"] = timeout
    if name is not None:
        parallel["name"] = name
    return {"parallel": parallel}


def when_node(
    *,
    predicate: str | Mapping[str, Any] | bool,
    then: Sequence[Mapping[str, Any]],
    else_: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    when: dict[str, Any] = {"predicate": predicate, "then": [dict(n) for n in then]}
    if else_ is not None:
        when["else"] = [dict(n) for n in else_]
    return {"when": when}


@dataclass(slots=True)
class PipelineSpec:
    """A mutable pipeline spec that can be compiled and exported."""

    nodes: list[dict[str, Any]] = field(default_factory=list)

    def add(self, node: Mapping[str, Any]) -> Self:
        self.nodes.append(dict(node))
        return self

    def add_type(self, type_name: str, config: Mapping[str, Any] | None = None) -> Self:
        return self.add(type_node(type_name, config))

    def add_chain(self, nodes: Sequence[Mapping[str, Any]]) -> Self:
        return self.add(chain_node(nodes))

    def add_parallel(
        self,
        nodes: Sequence[Mapping[str, Any]],
        *,
        strategy: str | AggregationStrategy | None = None,
        timeout: float | None = None,
        name: str | None = None,
    ) -> Self:
        return self.add(parallel_node(nodes, strategy=strategy, timeout=timeout, name=name))

    def add_when(
        self,
        *,
        predicate: str | Mapping[str, Any] | bool,
        then: Sequence[Mapping[str, Any]],
        else_: Sequence[Mapping[str, Any]] | None = None,
    ) -> Self:
        return self.add(when_node(predicate=predicate, then=then, else_=else_))

    def to_config(self) -> list[dict[str, Any]]:
        return [dict(n) for n in self.nodes]

    def dump(self, *, format: str = "json") -> str:
        fmt = _normalize_format(format)
        if fmt == "json":
            return json.dumps(self.to_config(), indent=2, sort_keys=True)
        return _dump_yaml(self.to_config())

    def save(self, path: str | Path, *, format: str | None = None) -> None:
        config_path = Path(path)
        fmt = _detect_format(format=format, path=config_path)
        config_path.write_text(self.dump(format=fmt), encoding="utf-8")

    def compile(self, compiler: MiddlewarePipelineCompiler[Any]) -> list[AgentMiddleware[Any]]:
        return compiler.compile_list(self.to_config())


def _dump_yaml(data: Any) -> str:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MiddlewareConfigError("YAML support requires PyYAML") from exc
    return yaml.safe_dump(data, sort_keys=True)


def _detect_format(*, format: str | None, path: Path | None) -> str:
    if format:
        return _normalize_format(format)

    if path is not None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return "json"
        if suffix in {".yaml", ".yml"}:
            return "yaml"

    raise MiddlewareConfigError("Unable to determine config format")


def _normalize_format(format: str) -> str:
    fmt = format.lower()
    if fmt == "json":
        return "json"
    if fmt in {"yaml", "yml"}:
        return "yaml"
    raise MiddlewareConfigError(f"Unknown config format: {format!r}")

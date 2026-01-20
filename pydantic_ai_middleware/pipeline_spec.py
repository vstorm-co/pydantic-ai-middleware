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
    """A mutable pipeline spec builder for defining middleware pipelines in Python.

    PipelineSpec provides a fluent API for building middleware pipelines that can be:
    - Compiled into middleware instances using a :class:`MiddlewarePipelineCompiler`
    - Exported as portable JSON/YAML config files for configuration-driven pipelines

    The builder uses the same skeleton nodes as config loading: `type`, `chain`,
    `parallel`, and `when`.

    Example:
        ```python
        from pydantic_ai_middleware import PipelineSpec
        from pydantic_ai_middleware.builder import MiddlewarePipelineCompiler

        # Build a pipeline spec
        spec = (
            PipelineSpec()
            .add_type("logging", {"level": "DEBUG"})
            .add_type("rate_limit", {"max_requests": 100})
            .add_when(
                predicate="is_admin",
                then=[{"type": "admin_audit"}],
                else_=[{"type": "user_audit"}],
            )
        )

        # Export to YAML
        spec.save("pipeline.yaml")

        # Or compile to middleware
        compiler = MiddlewarePipelineCompiler(registry)
        middleware = spec.compile(compiler)
        ```
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    """The list of pipeline nodes."""

    def add(self, node: Mapping[str, Any]) -> Self:
        """Add a raw node to the pipeline.

        Args:
            node: A node dictionary (e.g., {"type": "logging"}).

        Returns:
            Self for method chaining.
        """
        self.nodes.append(dict(node))
        return self

    def add_type(self, type_name: str, config: Mapping[str, Any] | None = None) -> Self:
        """Add a middleware type node to the pipeline.

        Args:
            type_name: The registered middleware type name.
            config: Optional configuration to pass to the middleware factory.

        Returns:
            Self for method chaining.

        Example:
            ```python
            spec.add_type("logging", {"level": "DEBUG"})
            ```
        """
        return self.add(type_node(type_name, config))

    def add_chain(self, nodes: Sequence[Mapping[str, Any]]) -> Self:
        """Add a chain node (sequential middleware execution).

        Args:
            nodes: Sequence of nodes to execute in order.

        Returns:
            Self for method chaining.

        Example:
            ```python
            spec.add_chain([
                {"type": "auth"},
                {"type": "logging"},
            ])
            ```
        """
        return self.add(chain_node(nodes))

    def add_parallel(
        self,
        nodes: Sequence[Mapping[str, Any]],
        *,
        strategy: str | AggregationStrategy | None = None,
        timeout: float | None = None,
        name: str | None = None,
    ) -> Self:
        """Add a parallel node (concurrent middleware execution).

        Args:
            nodes: Sequence of nodes to execute in parallel.
            strategy: Aggregation strategy for combining results.
                Options: "first", "last", "merge", "all".
            timeout: Optional timeout in seconds for parallel execution.
            name: Optional name for the parallel group.

        Returns:
            Self for method chaining.

        Example:
            ```python
            spec.add_parallel(
                [{"type": "cache"}, {"type": "metrics"}],
                strategy="merge",
                timeout=5.0,
            )
            ```
        """
        return self.add(parallel_node(nodes, strategy=strategy, timeout=timeout, name=name))

    def add_when(
        self,
        *,
        predicate: str | Mapping[str, Any] | bool,
        then: Sequence[Mapping[str, Any]],
        else_: Sequence[Mapping[str, Any]] | None = None,
    ) -> Self:
        """Add a conditional node (branching based on predicate).

        Args:
            predicate: Condition for branching. Can be:
                - A string (registered predicate name)
                - A dict (predicate with config)
                - A boolean (static condition)
            then: Nodes to execute when predicate is True.
            else_: Optional nodes to execute when predicate is False.

        Returns:
            Self for method chaining.

        Example:
            ```python
            spec.add_when(
                predicate="is_authenticated",
                then=[{"type": "user_middleware"}],
                else_=[{"type": "guest_middleware"}],
            )
            ```
        """
        return self.add(when_node(predicate=predicate, then=then, else_=else_))

    def to_config(self) -> list[dict[str, Any]]:
        """Convert the spec to a config-compatible list of nodes.

        Returns:
            A list of node dictionaries suitable for JSON/YAML serialization.
        """
        return [dict(n) for n in self.nodes]

    def dump(self, *, format: str = "json") -> str:
        """Serialize the pipeline to a string.

        Args:
            format: Output format, either "json" or "yaml".

        Returns:
            The serialized pipeline as a string.

        Raises:
            MiddlewareConfigError: If format is unknown or YAML requested but
                PyYAML is not installed.
        """
        fmt = _normalize_format(format)
        if fmt == "json":
            return json.dumps(self.to_config(), indent=2, sort_keys=True)
        return _dump_yaml(self.to_config())

    def save(self, path: str | Path, *, format: str | None = None) -> None:
        """Save the pipeline to a file.

        Args:
            path: File path to save to. Format is auto-detected from
                extension (.json, .yaml, .yml) if not specified.
            format: Optional explicit format ("json" or "yaml").

        Raises:
            MiddlewareConfigError: If format cannot be determined or is unknown.
        """
        config_path = Path(path)
        fmt = _detect_format(format=format, path=config_path)
        config_path.write_text(self.dump(format=fmt), encoding="utf-8")

    def compile(self, compiler: MiddlewarePipelineCompiler[Any]) -> list[AgentMiddleware[Any]]:
        """Compile the spec into middleware instances.

        Args:
            compiler: A MiddlewarePipelineCompiler with registered factories.

        Returns:
            A list of middleware instances ready to use with an agent.

        Raises:
            MiddlewareConfigError: If compilation fails (e.g., unknown types).
        """
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

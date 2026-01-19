"""Compile middleware pipelines from config specs.

This module intentionally keeps the pipeline "skeleton" nodes fixed:
- ``type``: instantiate a registered middleware
- ``chain``: sequential list of middleware
- ``parallel``: execute middleware in parallel (see :class:`ParallelMiddleware`)
- ``when``: conditional branching (see :class:`ConditionalMiddleware`)

User extensibility happens via a registry for middleware factories and predicates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from .base import AgentMiddleware, DepsT
from .conditional import ConditionalMiddleware, Predicate, PredicateFactory
from .exceptions import MiddlewareConfigError
from .strategies import AggregationStrategy

MiddlewareFactory = Callable[..., AgentMiddleware[DepsT]]

NodeHandler = Callable[
    ["MiddlewarePipelineCompiler[Any]", Mapping[str, Any]],
    AgentMiddleware[Any] | list[AgentMiddleware[Any]],
]

TDeps = TypeVar("TDeps")


@dataclass(slots=True)
class MiddlewareRegistry(Generic[TDeps]):
    """Registry of middleware factories and predicate functions.

    This is the user-facing extension mechanism: add new middleware types and
    predicates without modifying the compiler.
    """

    middleware: dict[str, MiddlewareFactory[TDeps]] = field(default_factory=dict)
    predicates: dict[str, PredicateFactory | Predicate] = field(default_factory=dict)

    def register_middleware(
        self,
        name: str,
        factory: MiddlewareFactory[TDeps],
        *,
        overwrite: bool = False,
    ) -> MiddlewareFactory[TDeps]:
        """Register a middleware factory under a name."""
        if not overwrite and name in self.middleware:
            raise MiddlewareConfigError(f"Middleware '{name}' is already registered")
        self.middleware[name] = factory
        return factory

    def middleware_factory(
        self,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ) -> Callable[[MiddlewareFactory[TDeps]], MiddlewareFactory[TDeps]]:
        """Decorator for registering a middleware factory."""

        def _decorator(factory: MiddlewareFactory[TDeps]) -> MiddlewareFactory[TDeps]:
            key = name or factory.__name__
            return self.register_middleware(key, factory, overwrite=overwrite)

        return _decorator

    def register_predicate(
        self,
        name: str,
        predicate: PredicateFactory | Predicate,
        *,
        overwrite: bool = False,
    ) -> PredicateFactory | Predicate:
        """Register a predicate (or predicate factory) under a name."""
        if not overwrite and name in self.predicates:
            raise MiddlewareConfigError(f"Predicate '{name}' is already registered")
        self.predicates[name] = predicate
        return predicate

    def predicate(
        self,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ) -> Callable[[PredicateFactory | Predicate], PredicateFactory | Predicate]:
        """Decorator for registering a predicate."""

        def _decorator(item: PredicateFactory | Predicate) -> PredicateFactory | Predicate:
            key = name or getattr(item, "__name__", None)
            if not key:
                raise MiddlewareConfigError("Predicate name is required for registration")
            return self.register_predicate(key, item, overwrite=overwrite)

        return _decorator


class MiddlewarePipelineCompiler(Generic[DepsT]):
    """Compile config specs into middleware instances."""

    def __init__(
        self,
        *,
        registry: MiddlewareRegistry[DepsT],
        node_handlers: Mapping[str, NodeHandler] | None = None,
    ) -> None:
        self._registry = registry
        self._node_handlers: dict[str, NodeHandler] = dict(node_handlers or DEFAULT_NODE_HANDLERS)

    def compile(
        self,
        config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]]:
        """Compile a spec mapping or a list of spec mappings."""
        if not isinstance(config, (Mapping, Sequence)) or isinstance(config, str):
            raise MiddlewareConfigError(
                f"Middleware spec must be a mapping or sequence, got {type(config).__name__}"
            )

        if isinstance(config, Sequence) and not isinstance(config, Mapping):
            return self._compile_list(config)

        return self._compile_node(config)

    def compile_list(
        self,
        config: Mapping[str, Any] | Sequence[Mapping[str, Any]] | AgentMiddleware[DepsT],
    ) -> list[AgentMiddleware[DepsT]]:
        """Compile a config into a list of middleware."""
        if isinstance(config, AgentMiddleware):
            return [config]

        if isinstance(config, Sequence) and not isinstance(config, (str, Mapping)):
            return self._compile_list(config)

        if isinstance(config, Mapping):
            built = self.compile(config)
            if isinstance(built, list):
                return built
            return [built]

        raise MiddlewareConfigError(
            "Middleware config must be a mapping, sequence, or AgentMiddleware, "
            f"got {type(config).__name__}"
        )

    def register_node_handler(
        self,
        key: str,
        handler: NodeHandler,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a new top-level node handler (advanced extension point)."""
        if not overwrite and key in self._node_handlers:
            raise MiddlewareConfigError(f"Node handler for '{key}' is already registered")
        self._node_handlers[key] = handler

    def _compile_list(self, specs: Sequence[Mapping[str, Any]]) -> list[AgentMiddleware[DepsT]]:
        result: list[AgentMiddleware[DepsT]] = []
        for spec in specs:
            built = self._compile_node(spec)
            if isinstance(built, list):
                result.extend(built)
            else:
                result.append(built)
        return result

    def _compile_node(
        self, spec: Mapping[str, Any]
    ) -> AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]]:
        if not isinstance(spec, Mapping):
            raise MiddlewareConfigError(
                f"Middleware spec must be a mapping, got {type(spec).__name__}"
            )

        matching_keys = [key for key in self._node_handlers if key in spec]
        if not matching_keys:
            valid = ", ".join(self._node_handlers.keys())
            raise MiddlewareConfigError(
                f"Unknown middleware spec: expected one of {valid}, got keys {list(spec.keys())}"
            )
        if len(matching_keys) > 1:
            raise MiddlewareConfigError(
                "Ambiguous middleware spec: expected exactly one of "
                f"{matching_keys}, got keys {list(spec.keys())}"
            )

        handler_key = matching_keys[0]
        handler = self._node_handlers[handler_key]
        built_any = handler(cast(MiddlewarePipelineCompiler[Any], self), spec)
        return cast(AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]], built_any)

    def _parse_strategy(self, value: str | AggregationStrategy | None) -> AggregationStrategy:
        if value is None:
            return AggregationStrategy.ALL_MUST_PASS
        if isinstance(value, AggregationStrategy):
            return value
        if isinstance(value, str):
            try:
                return AggregationStrategy(value)
            except ValueError as exc:
                valid = [s.value for s in AggregationStrategy]
                raise MiddlewareConfigError(
                    f"Unknown aggregation strategy: {value!r}. Valid options: {valid}"
                ) from exc
        raise MiddlewareConfigError(
            "Invalid aggregation strategy value: expected str or AggregationStrategy, "
            f"got {type(value).__name__}"
        )

    def _build_predicate(
        self, spec: str | Mapping[str, Any] | Callable[..., bool] | bool
    ) -> Predicate:
        if callable(spec):
            return spec  # type: ignore[return-value]

        if isinstance(spec, bool):
            return lambda _: spec

        if isinstance(spec, str):
            if spec not in self._registry.predicates:
                raise MiddlewareConfigError(f"Unknown predicate: {spec!r}")
            pred = self._registry.predicates[spec]
            if callable(pred):
                return pred  # type: ignore[return-value]
            raise MiddlewareConfigError(f"Predicate {spec!r} is not callable")

        if isinstance(spec, Mapping):
            name = spec.get("name")
            if not name:
                raise MiddlewareConfigError("Predicate config missing 'name' key")
            if name not in self._registry.predicates:
                raise MiddlewareConfigError(f"Unknown predicate: {name!r}")

            factory = self._registry.predicates[name]
            config = spec.get("config", {})
            if not isinstance(config, Mapping):
                raise MiddlewareConfigError(
                    f"Predicate config must be a mapping, got {type(config).__name__}"
                )
            return factory(**config)  # type: ignore[call-arg,return-value]

        raise MiddlewareConfigError(
            "Invalid predicate specification: expected str, dict, callable, or bool, "
            f"got {type(spec).__name__}"
        )


def _handle_type(
    compiler: MiddlewarePipelineCompiler[Any], spec: Mapping[str, Any]
) -> AgentMiddleware[Any]:
    type_name = spec["type"]
    if not isinstance(type_name, str):
        raise MiddlewareConfigError(f"type must be a string, got {type(type_name).__name__}")

    if type_name not in compiler._registry.middleware:
        raise MiddlewareConfigError(
            f"Unknown middleware type: {type_name!r}. Register it in the registry."
        )

    factory = compiler._registry.middleware[type_name]
    config = spec.get("config", {})
    if not isinstance(config, Mapping):
        raise MiddlewareConfigError(f"config must be a mapping, got {type(config).__name__}")

    try:
        return factory(**config)
    except TypeError as exc:
        raise MiddlewareConfigError(f"Failed to construct middleware '{type_name}': {exc}") from exc


def _handle_chain(
    compiler: MiddlewarePipelineCompiler[Any], spec: Mapping[str, Any]
) -> list[AgentMiddleware[Any]]:
    chain_spec = spec["chain"]
    if not isinstance(chain_spec, Sequence) or isinstance(chain_spec, str):
        raise MiddlewareConfigError(f"chain must be a sequence, got {type(chain_spec).__name__}")
    return compiler._compile_list(chain_spec)


def _handle_parallel(
    compiler: MiddlewarePipelineCompiler[Any], spec: Mapping[str, Any]
) -> AgentMiddleware[Any]:
    from .parallel import ParallelMiddleware

    parallel_spec = spec["parallel"]
    if not isinstance(parallel_spec, Mapping):
        raise MiddlewareConfigError(
            f"parallel spec must be a mapping, got {type(parallel_spec).__name__}"
        )

    middleware_specs = parallel_spec.get("middleware", [])
    if not isinstance(middleware_specs, Sequence) or isinstance(middleware_specs, str):
        raise MiddlewareConfigError(
            f"parallel.middleware must be a sequence, got {type(middleware_specs).__name__}"
        )

    middleware_list = compiler._compile_list(middleware_specs)

    return ParallelMiddleware(
        middleware=middleware_list,
        strategy=compiler._parse_strategy(parallel_spec.get("strategy")),
        timeout=parallel_spec.get("timeout"),
        name=parallel_spec.get("name"),
    )


def _handle_when(
    compiler: MiddlewarePipelineCompiler[Any], spec: Mapping[str, Any]
) -> AgentMiddleware[Any]:
    when_spec = spec["when"]
    if not isinstance(when_spec, Mapping):
        raise MiddlewareConfigError(f"when spec must be a mapping, got {type(when_spec).__name__}")

    predicate_spec = when_spec.get("predicate")
    if predicate_spec is None:
        raise MiddlewareConfigError("Invalid predicate specification")
    predicate = compiler._build_predicate(predicate_spec)

    then_specs = when_spec.get("then", [])
    if not isinstance(then_specs, Sequence) or isinstance(then_specs, str):
        raise MiddlewareConfigError(
            f"when.then must be a sequence, got {type(then_specs).__name__}"
        )

    else_specs = when_spec.get("else")
    if else_specs is not None and (
        not isinstance(else_specs, Sequence) or isinstance(else_specs, str)
    ):
        raise MiddlewareConfigError(
            f"when.else must be a sequence, got {type(else_specs).__name__}"
        )

    then_list = compiler._compile_list(then_specs)
    else_list = compiler._compile_list(else_specs) if else_specs else None

    return ConditionalMiddleware(
        condition=predicate,
        when_true=then_list,
        when_false=else_list,
    )


DEFAULT_NODE_HANDLERS: dict[str, NodeHandler] = {
    "type": _handle_type,
    "chain": _handle_chain,
    "parallel": _handle_parallel,
    "when": _handle_when,
}


def build_middleware(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
) -> AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]]:
    """Backwards-compatible wrapper: compile middleware from config."""
    reg = MiddlewareRegistry[DepsT](middleware=dict(registry), predicates=dict(predicates or {}))
    compiler = MiddlewarePipelineCompiler(registry=reg)
    return compiler.compile(config)


def build_middleware_list(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]] | AgentMiddleware[DepsT],
    *,
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
) -> list[AgentMiddleware[DepsT]]:
    """Backwards-compatible wrapper: compile middleware list from config."""
    reg = MiddlewareRegistry[DepsT](middleware=dict(registry), predicates=dict(predicates or {}))
    compiler = MiddlewarePipelineCompiler(registry=reg)
    return compiler.compile_list(config)

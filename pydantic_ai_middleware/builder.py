"""Helpers for composing and configuring middleware."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .base import AgentMiddleware, DepsT
from .conditional import ConditionalMiddleware, Predicate, PredicateFactory
from .exceptions import MiddlewareConfigError
from .strategies import AggregationStrategy

MiddlewareFactory = Callable[..., AgentMiddleware[DepsT]]


def build_middleware(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
) -> AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]]:
    """Build a single middleware from a config spec."""
    if not isinstance(config, (Mapping, Sequence)) or isinstance(config, str):
        raise MiddlewareConfigError(
            f"Middleware spec must be a mapping or sequence, got {type(config).__name__}"
        )

    if isinstance(config, Sequence) and not isinstance(config, Mapping):
        middleware_list = _build_middleware_list_from_specs(config, registry, predicates or {})
        return middleware_list

    return _build_single_middleware(config, registry, predicates or {})


def build_middleware_list(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]] | AgentMiddleware[DepsT],
    *,
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
) -> list[AgentMiddleware[DepsT]]:
    """Build a list of middleware from config."""
    if isinstance(config, AgentMiddleware):
        return [config]

    if isinstance(config, Sequence) and not isinstance(config, (str, Mapping)):
        return _build_middleware_list_from_specs(config, registry, predicates or {})

    if isinstance(config, Mapping):
        built = build_middleware(config, registry=registry, predicates=predicates)
        if isinstance(built, list):
            return built
        return [built]

    raise MiddlewareConfigError(
        "Middleware config must be a mapping, sequence, or AgentMiddleware, "
        f"got {type(config).__name__}"
    )


def _parse_strategy(value: str | AggregationStrategy | None) -> AggregationStrategy:
    """Parse aggregation strategy from string or enum."""
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
    spec: str | Mapping[str, Any] | Callable[..., bool] | bool,
    predicates: Mapping[str, PredicateFactory | Predicate],
) -> Predicate:
    """Build a predicate from a spec."""
    if callable(spec):
        return spec  # type: ignore[return-value]

    if isinstance(spec, bool):
        return lambda _: spec

    if isinstance(spec, str):
        if spec not in predicates:
            raise MiddlewareConfigError(f"Unknown predicate: {spec!r}")
        pred = predicates[spec]
        if callable(pred):
            return pred  # type: ignore[return-value]
        raise MiddlewareConfigError(f"Predicate {spec!r} is not callable")

    if isinstance(spec, Mapping):
        name = spec.get("name")
        if not name:
            raise MiddlewareConfigError("Predicate config missing 'name' key")
        if name not in predicates:
            raise MiddlewareConfigError(f"Unknown predicate: {name!r}")

        factory = predicates[name]
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


def _build_middleware_list_from_specs(
    specs: Sequence[Mapping[str, Any]],
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate] | None = None,
) -> list[AgentMiddleware[DepsT]]:
    """Build a list of middleware from specs."""
    result: list[AgentMiddleware[DepsT]] = []
    for spec in specs:
        built = _build_single_middleware(spec, registry, predicates or {})
        if isinstance(built, list):
            result.extend(built)
        else:
            result.append(built)
    return result


def _build_chain(
    spec: Mapping[str, Any],
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate],
) -> list[AgentMiddleware[DepsT]]:
    """Build middleware list from a chain spec."""
    chain_spec = spec["chain"]
    if not isinstance(chain_spec, Sequence) or isinstance(chain_spec, str):
        raise MiddlewareConfigError(f"chain must be a sequence, got {type(chain_spec).__name__}")
    return _build_middleware_list_from_specs(chain_spec, registry, predicates)


def _build_parallel(
    spec: Mapping[str, Any],
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate],
) -> AgentMiddleware[DepsT]:
    """Build parallel middleware from a parallel spec."""
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

    middleware_list = _build_middleware_list_from_specs(middleware_specs, registry, predicates)

    return ParallelMiddleware(
        middleware=middleware_list,
        strategy=_parse_strategy(parallel_spec.get("strategy")),
        timeout=parallel_spec.get("timeout"),
        name=parallel_spec.get("name"),
    )


def _build_when(
    spec: Mapping[str, Any],
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate],
) -> AgentMiddleware[DepsT]:
    """Build conditional middleware from a when spec."""
    when_spec = spec["when"]
    if not isinstance(when_spec, Mapping):
        raise MiddlewareConfigError(f"when spec must be a mapping, got {type(when_spec).__name__}")

    predicate_spec = when_spec.get("predicate")
    if predicate_spec is None:
        raise MiddlewareConfigError("Invalid predicate specification")

    predicate = _build_predicate(predicate_spec, predicates)

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

    then_list = _build_middleware_list_from_specs(then_specs, registry, predicates)
    else_list = (
        _build_middleware_list_from_specs(else_specs, registry, predicates) if else_specs else None
    )

    return ConditionalMiddleware(
        condition=predicate,
        when_true=then_list,
        when_false=else_list,
    )


def _build_type(
    spec: Mapping[str, Any],
    registry: Mapping[str, MiddlewareFactory[DepsT]],
) -> AgentMiddleware[DepsT]:
    """Build middleware from a type spec."""
    type_name = spec["type"]
    if not isinstance(type_name, str):
        raise MiddlewareConfigError(f"type must be a string, got {type(type_name).__name__}")

    if type_name not in registry:
        raise MiddlewareConfigError(
            f"Unknown middleware type: {type_name!r}. Register it in the registry."
        )

    factory = registry[type_name]
    config = spec.get("config", {})

    if not isinstance(config, Mapping):
        raise MiddlewareConfigError(f"config must be a mapping, got {type(config).__name__}")

    try:
        return factory(**config)
    except TypeError as exc:
        raise MiddlewareConfigError(f"Failed to construct middleware '{type_name}': {exc}") from exc


def _build_single_middleware(
    spec: Mapping[str, Any],
    registry: Mapping[str, MiddlewareFactory[DepsT]],
    predicates: Mapping[str, PredicateFactory | Predicate],
) -> AgentMiddleware[DepsT] | list[AgentMiddleware[DepsT]]:
    """Build a single middleware from a spec dict."""
    if not isinstance(spec, Mapping):
        raise MiddlewareConfigError(f"Middleware spec must be a mapping, got {type(spec).__name__}")

    if "chain" in spec:
        return _build_chain(spec, registry, predicates)

    if "parallel" in spec:
        return _build_parallel(spec, registry, predicates)

    if "when" in spec:
        return _build_when(spec, registry, predicates)

    if "type" in spec:
        return _build_type(spec, registry)

    raise MiddlewareConfigError(
        "Unknown middleware spec: expected 'type', 'chain', 'parallel', or 'when' key, "
        f"got keys {list(spec.keys())}"
    )

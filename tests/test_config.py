"""Tests for config loaders and builders."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from pydantic_ai_middleware import AgentMiddleware
from pydantic_ai_middleware.builder import (
    MiddlewarePipelineCompiler,
    MiddlewareRegistry,
    build_middleware,
    build_middleware_list,
)
from pydantic_ai_middleware.config_loaders import (
    _detect_format,
    load_middleware_config_path,
    load_middleware_config_text,
)
from pydantic_ai_middleware.exceptions import MiddlewareConfigError


class DummyMiddleware(AgentMiddleware[None]):
    """Simple middleware for config tests."""

    def __init__(self, **config: Any) -> None:
        self.config: dict[str, Any] = config


class FailingMiddleware(AgentMiddleware[None]):
    """Middleware that requires a positional arg (to trigger TypeError)."""

    def __init__(self, required: int, **config: Any) -> None:
        self.required = required
        self.config: dict[str, Any] = config


def test_load_json_list() -> None:
    registry = {"dummy": DummyMiddleware}
    text = '[{"type": "dummy", "config": {"id": 1}}, {"type": "dummy", "config": {"id": 2}}]'
    result = load_middleware_config_text(text, registry=registry, format="json")

    assert len(result) == 2
    mw0 = result[0]
    mw1 = result[1]
    assert isinstance(mw0, DummyMiddleware)
    assert isinstance(mw1, DummyMiddleware)
    assert mw0.config == {"id": 1}
    assert mw1.config == {"id": 2}


def test_load_chain_node() -> None:
    registry = {"dummy": DummyMiddleware}
    text = (
        '[{"chain": ['
        '{"type": "dummy", "config": {"id": 1}}, '
        '{"type": "dummy", "config": {"id": 2}}'
        "]}]"
    )
    result = load_middleware_config_text(text, registry=registry, format="json")

    assert len(result) == 2
    mw0 = result[0]
    mw1 = result[1]
    assert isinstance(mw0, DummyMiddleware)
    assert isinstance(mw1, DummyMiddleware)
    assert mw0.config == {"id": 1}
    assert mw1.config == {"id": 2}


def test_chain_invalid_type_raises() -> None:
    registry = {"dummy": DummyMiddleware}
    text = '[{"chain": "not a list"}]'
    with pytest.raises(MiddlewareConfigError, match="chain must be a sequence"):
        load_middleware_config_text(text, registry=registry, format="json")


def test_when_predicate_string() -> None:
    registry = {"dummy": DummyMiddleware}
    predicates = {"always": lambda ctx: True}
    text = '[{"when": {"predicate": "always", "then": [{"type": "dummy"}]}}]'
    result = load_middleware_config_text(
        text, registry=registry, predicates=predicates, format="json"
    )

    assert len(result) == 1


def test_format_detection_by_content_json() -> None:
    registry = {"dummy": DummyMiddleware}
    text = '[{"type": "dummy"}]'
    result = load_middleware_config_text(text, registry=registry)
    assert len(result) == 1


def test_load_yaml_when_available() -> None:
    pytest.importorskip("yaml", reason="PyYAML not installed")
    registry = {"dummy": DummyMiddleware}
    text = "- type: dummy\n  config:\n    id: 1\n"
    result = load_middleware_config_text(text, registry=registry)

    assert len(result) == 1
    mw0 = result[0]
    assert isinstance(mw0, DummyMiddleware)
    assert mw0.config == {"id": 1}


def test_load_yaml_path_when_available(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="PyYAML not installed")
    registry = {"dummy": DummyMiddleware}
    path = tmp_path / "pipeline.yaml"
    path.write_text("- type: dummy\n  config:\n    id: 2\n", encoding="utf-8")

    result = load_middleware_config_path(path, registry=registry)

    assert len(result) == 1
    assert isinstance(result[0], DummyMiddleware)
    assert result[0].config == {"id": 2}


def test_invalid_json_raises() -> None:
    registry = {"dummy": DummyMiddleware}
    with pytest.raises(MiddlewareConfigError, match="Invalid JSON"):
        load_middleware_config_text("{bad json", registry=registry, format="json")


def test_unknown_format_raises() -> None:
    with pytest.raises(MiddlewareConfigError, match="Unknown config format"):
        load_middleware_config_text("- type: dummy", registry={}, format="toml")


def test_load_yaml_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = {"dummy": DummyMiddleware}

    monkeypatch.setitem(sys.modules, "yaml", None)  # Simulates missing module

    with pytest.raises(MiddlewareConfigError, match="YAML support requires PyYAML"):
        load_middleware_config_text("- type: dummy", registry=registry, format="yaml")


def test_detect_format_missing_inputs_raises() -> None:
    with pytest.raises(MiddlewareConfigError, match="Unable to determine config format"):
        _detect_format()


def test_detect_format_path_yaml() -> None:
    assert _detect_format(path=Path("pipeline.yaml")) == "yaml"


def test_load_path_unknown_suffix_uses_text(tmp_path: Path) -> None:
    registry = {"dummy": DummyMiddleware}
    path = tmp_path / "pipeline.txt"
    path.write_text('[{"type": "dummy", "config": {"id": 3}}]', encoding="utf-8")

    result = load_middleware_config_path(path, registry=registry)

    assert len(result) == 1
    assert isinstance(result[0], DummyMiddleware)
    assert result[0].config == {"id": 3}


def test_builder_type_errors() -> None:
    registry = {"dummy": DummyMiddleware}
    with pytest.raises(MiddlewareConfigError, match="Middleware spec must be a mapping"):
        build_middleware(123, registry=registry)  # type: ignore[arg-type]

    with pytest.raises(MiddlewareConfigError, match="Unknown middleware type"):
        build_middleware({"type": "missing"}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="type must be a string"):
        build_middleware({"type": 123}, registry=registry)  # type: ignore[arg-type]

    with pytest.raises(MiddlewareConfigError, match="config must be a mapping"):
        build_middleware({"type": "dummy", "config": "bad"}, registry=registry)


def test_builder_factory_typeerror() -> None:
    registry = {"fail": FailingMiddleware}
    with pytest.raises(MiddlewareConfigError, match="Failed to construct middleware"):
        build_middleware({"type": "fail", "config": {"x": 1}}, registry=registry)


def test_builder_parallel_and_when_errors() -> None:
    registry = {"dummy": DummyMiddleware}

    with pytest.raises(MiddlewareConfigError, match="parallel spec must be a mapping"):
        build_middleware({"parallel": "bad"}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="parallel.middleware must be a sequence"):
        build_middleware({"parallel": {"middleware": "bad"}}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="when spec must be a mapping"):
        build_middleware({"when": "bad"}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="Invalid predicate specification"):
        build_middleware({"when": {"then": []}}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="when.then must be a sequence"):
        build_middleware({"when": {"predicate": True, "then": "bad"}}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="when.else must be a sequence"):
        build_middleware(
            {"when": {"predicate": True, "then": [], "else": "bad"}}, registry=registry
        )


def test_builder_sequence_input() -> None:
    registry = {"dummy": DummyMiddleware}
    specs = [{"type": "dummy"}, {"type": "dummy"}]
    result = build_middleware(specs, registry=registry)

    assert isinstance(result, list)
    assert len(result) == 2


def test_builder_sequence_item_error() -> None:
    registry = {"dummy": DummyMiddleware}
    with pytest.raises(MiddlewareConfigError, match="Middleware spec must be a mapping"):
        build_middleware([{"type": "dummy"}, "bad"], registry=registry)  # type: ignore[list-item]


def test_builder_unknown_spec_key() -> None:
    registry = {"dummy": DummyMiddleware}
    with pytest.raises(MiddlewareConfigError, match="Unknown middleware spec"):
        build_middleware({"unknown": True}, registry=registry)


def test_builder_parallel_strategy_variants() -> None:
    from pydantic_ai_middleware.strategies import AggregationStrategy

    registry = {"dummy": DummyMiddleware}

    result = build_middleware(
        {"parallel": {"middleware": [{"type": "dummy"}]}},
        registry=registry,
    )
    assert isinstance(result, AgentMiddleware)

    result = build_middleware(
        {
            "parallel": {
                "middleware": [{"type": "dummy"}],
                "strategy": AggregationStrategy.FIRST_SUCCESS,
            }
        },
        registry=registry,
    )
    assert isinstance(result, AgentMiddleware)

    result = build_middleware(
        {"parallel": {"middleware": [{"type": "dummy"}], "strategy": "race"}},
        registry=registry,
    )
    assert isinstance(result, AgentMiddleware)

    with pytest.raises(MiddlewareConfigError, match="Unknown aggregation strategy"):
        build_middleware(
            {"parallel": {"middleware": [{"type": "dummy"}], "strategy": "bad"}},
            registry=registry,
        )

    with pytest.raises(MiddlewareConfigError, match="Invalid aggregation strategy value"):
        build_middleware(
            {"parallel": {"middleware": [{"type": "dummy"}], "strategy": 123}},
            registry=registry,
        )


def test_builder_predicates() -> None:
    registry = {"dummy": DummyMiddleware}

    def pred_factory(threshold: int) -> Any:
        return lambda ctx: threshold > 0

    predicates = {
        "always": lambda ctx: True,
        "factory": pred_factory,
    }

    result = build_middleware(
        {"when": {"predicate": "always", "then": [{"type": "dummy"}]}},
        registry=registry,
        predicates=predicates,
    )
    assert isinstance(result, AgentMiddleware)

    result = build_middleware(
        {"when": {"predicate": {"name": "factory", "config": {"threshold": 1}}, "then": []}},
        registry=registry,
        predicates=predicates,
    )
    assert isinstance(result, AgentMiddleware)

    with pytest.raises(MiddlewareConfigError, match="Unknown predicate"):
        build_middleware(
            {"when": {"predicate": "missing", "then": []}},
            registry=registry,
            predicates=predicates,
        )

    with pytest.raises(MiddlewareConfigError, match="Predicate config missing 'name'"):
        build_middleware({"when": {"predicate": {"config": {}}, "then": []}}, registry=registry)

    with pytest.raises(MiddlewareConfigError, match="Predicate config must be a mapping"):
        build_middleware(
            {"when": {"predicate": {"name": "factory", "config": "bad"}, "then": []}},
            registry=registry,
            predicates=predicates,
        )

    with pytest.raises(MiddlewareConfigError, match="Predicate .* is not callable"):
        build_middleware(
            {"when": {"predicate": "always", "then": []}},
            registry=registry,
            predicates={"always": cast(Any, object())},
        )

    with pytest.raises(MiddlewareConfigError, match="Invalid predicate specification"):
        build_middleware(
            {"when": {"predicate": 123, "then": []}},
            registry=registry,
        )


def test_build_middleware_list_inputs() -> None:
    registry = {"dummy": DummyMiddleware}
    mw = DummyMiddleware()

    assert build_middleware_list(mw, registry=registry) == [mw]

    items = build_middleware_list([{"type": "dummy"}], registry=registry)
    assert len(items) == 1

    with pytest.raises(MiddlewareConfigError, match="Middleware config must be a mapping"):
        build_middleware_list(123, registry=registry)  # type: ignore[arg-type]

    result = build_middleware_list({"type": "dummy"}, registry=registry)
    assert len(result) == 1

    result = build_middleware_list({"chain": [{"type": "dummy"}]}, registry=registry)
    assert len(result) == 1


def test_builder_class_api() -> None:
    reg = MiddlewareRegistry()
    reg.register_middleware("dummy", DummyMiddleware)
    compiler = MiddlewarePipelineCompiler(registry=reg)

    result = compiler.compile({"type": "dummy", "config": {"id": 1}})
    assert isinstance(result, DummyMiddleware)
    assert result.config == {"id": 1}

    items = compiler.compile_list([{"type": "dummy"}, {"type": "dummy"}])
    assert len(items) == 2


def test_registry_decorators_and_overwrite() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry()

    @reg.middleware_factory()
    def Dummy(**config: Any) -> DummyMiddleware:
        return DummyMiddleware(**config)

    assert "Dummy" in reg.middleware

    @reg.predicate()
    def always(ctx: Any) -> bool:
        return True

    assert "always" in reg.predicates

    with pytest.raises(MiddlewareConfigError, match="already registered"):
        reg.register_middleware("Dummy", Dummy, overwrite=False)

    reg.register_middleware("Dummy", Dummy, overwrite=True)


def test_compiler_ambiguous_spec_raises() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry(middleware={"dummy": DummyMiddleware})
    compiler = MiddlewarePipelineCompiler(registry=reg)

    with pytest.raises(MiddlewareConfigError, match="Ambiguous middleware spec"):
        compiler.compile({"type": "dummy", "chain": []})


def test_compiler_when_uses_registry_predicates() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry(middleware={"dummy": DummyMiddleware})

    @reg.predicate("always_true")
    def always_true(ctx: Any) -> bool:
        return True

    compiler = MiddlewarePipelineCompiler(registry=reg)
    result = compiler.compile(
        {
            "when": {
                "predicate": "always_true",
                "then": [{"type": "dummy", "config": {"id": 1}}],
                "else": [{"type": "dummy", "config": {"id": 2}}],
            }
        }
    )

    from pydantic_ai_middleware.conditional import ConditionalMiddleware

    assert isinstance(result, ConditionalMiddleware)


def test_registry_register_predicate_duplicate_raises() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry()

    reg.register_predicate("p", lambda ctx: True)

    with pytest.raises(MiddlewareConfigError, match="Predicate 'p' is already registered"):
        reg.register_predicate("p", lambda ctx: False)


def test_registry_predicate_decorator_requires_name() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry()

    class CallablePredicate:
        def __call__(self, ctx: Any) -> bool:
            return True

    decorator = reg.predicate()
    with pytest.raises(MiddlewareConfigError, match="Predicate name is required for registration"):
        decorator(CallablePredicate())


def test_compiler_register_node_handler_duplicate_raises() -> None:
    reg = MiddlewareRegistry()
    reg.register_middleware("dummy", DummyMiddleware)
    compiler = MiddlewarePipelineCompiler(registry=reg)

    def handler(_: MiddlewarePipelineCompiler[Any], __: Mapping[str, Any]) -> Any:
        return DummyMiddleware()

    with pytest.raises(
        MiddlewareConfigError, match="Node handler for 'type' is already registered"
    ):
        compiler.register_node_handler("type", handler)


def test_compiler_register_node_handler_overwrite_allows_replacement() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry(middleware={"dummy": DummyMiddleware})
    compiler = MiddlewarePipelineCompiler(registry=reg)

    def handler(_: MiddlewarePipelineCompiler[Any], __: Mapping[str, Any]) -> Any:
        return DummyMiddleware()

    compiler.register_node_handler("type", handler, overwrite=True)


def test_register_helpers() -> None:
    from pydantic_ai_middleware.config_loaders import register_middleware, register_predicate

    registry: dict[str, Any] = {}
    predicates: dict[str, Callable[[Any], bool]] = {}

    @register_middleware(registry)
    class Registered(DummyMiddleware):
        pass

    assert "Registered" in registry

    register_middleware(registry, DummyMiddleware, name="dummy")
    assert "dummy" in registry

    with pytest.raises(MiddlewareConfigError, match="already registered"):
        register_middleware(registry, DummyMiddleware, name="dummy")

    @register_predicate(predicates)
    def predicate(ctx: Any) -> bool:
        return True

    assert "predicate" in predicates

    register_predicate(predicates, predicate, name="always")
    assert "always" in predicates

    with pytest.raises(MiddlewareConfigError, match="already registered"):
        register_predicate(predicates, predicate, name="always")

    class CallablePredicate:
        def __call__(self, ctx: Any) -> bool:
            return True

    with pytest.raises(MiddlewareConfigError, match="Predicate name is required"):
        register_predicate(predicates, CallablePredicate(), name=None)

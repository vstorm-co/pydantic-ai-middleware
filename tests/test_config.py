"""Tests for config loaders and builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pydantic_ai_middleware import AgentMiddleware
from pydantic_ai_middleware.builder import build_middleware, build_middleware_list
from pydantic_ai_middleware.config_loaders import (
    dump_middleware_config,
    load_middleware_config_path,
    load_middleware_config_text,
    save_middleware_config_path,
)
from pydantic_ai_middleware.exceptions import MiddlewareConfigError


class DummyMiddleware(AgentMiddleware[None]):
    """Simple middleware for config tests."""

    def __init__(self, **config: Any) -> None:
        self.config = config


class FailingMiddleware(AgentMiddleware[None]):
    """Middleware that requires a positional arg (to trigger TypeError)."""

    def __init__(self, required: int, **config: Any) -> None:
        self.required = required
        self.config = config


def test_load_json_list() -> None:
    registry = {"dummy": DummyMiddleware}
    text = '[{"type": "dummy", "config": {"id": 1}}, {"type": "dummy", "config": {"id": 2}}]'
    result = load_middleware_config_text(text, registry=registry, format="json")

    assert len(result) == 2
    assert result[0].config == {"id": 1}
    assert result[1].config == {"id": 2}


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
    assert result[0].config == {"id": 1}
    assert result[1].config == {"id": 2}


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


def test_dump_normalizes_chain_single_item() -> None:
    config: Any = [{"chain": {"type": "dummy"}}]
    result = dump_middleware_config(config, format="json")

    assert '"chain"' in result
    assert '"type": "dummy"' in result
    assert result.lstrip().startswith("[")


def test_save_and_load_json_path(tmp_path: Path) -> None:
    registry = {"dummy": DummyMiddleware}
    config = [{"type": "dummy", "config": {"id": 1}}]
    path = tmp_path / "pipeline.json"

    save_middleware_config_path(config, path)
    result = load_middleware_config_path(path, registry=registry)

    assert len(result) == 1
    assert result[0].config == {"id": 1}


def test_load_yaml_when_available() -> None:
    pytest.importorskip("yaml", reason="PyYAML not installed")
    registry = {"dummy": DummyMiddleware}
    text = "- type: dummy\n  config:\n    id: 1\n"
    result = load_middleware_config_text(text, registry=registry)

    assert len(result) == 1
    assert result[0].config == {"id": 1}


def test_dump_yaml_when_available() -> None:
    pytest.importorskip("yaml", reason="PyYAML not installed")
    config = [{"type": "dummy", "config": {"id": 1}}]
    result = dump_middleware_config(config, format="yaml")

    assert "type: dummy" in result


def test_save_yaml_path_when_available(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="PyYAML not installed")
    config = [{"type": "dummy"}]
    path = tmp_path / "pipeline.yml"

    save_middleware_config_path(config, path)
    assert path.read_text()


def test_invalid_json_raises() -> None:
    registry = {"dummy": DummyMiddleware}
    with pytest.raises(MiddlewareConfigError, match="Invalid JSON"):
        load_middleware_config_text("{bad json", registry=registry, format="json")


def test_unknown_format_raises() -> None:
    with pytest.raises(MiddlewareConfigError, match="Unknown config format"):
        dump_middleware_config([{"type": "dummy"}], format="toml")


def test_save_unknown_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "pipeline"
    with pytest.raises(MiddlewareConfigError, match="Unable to determine config format"):
        save_middleware_config_path([{"type": "dummy"}], path)


def test_load_yaml_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = {"dummy": DummyMiddleware}

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(MiddlewareConfigError, match="YAML support requires PyYAML"):
        load_middleware_config_text("- type: dummy", registry=registry, format="yaml")


def test_dump_yaml_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(MiddlewareConfigError, match="YAML support requires PyYAML"):
        dump_middleware_config([{"type": "dummy"}], format="yaml")


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
            predicates={"always": object()},
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


def test_register_helpers() -> None:
    from pydantic_ai_middleware.config_loaders import register_middleware, register_predicate

    registry: dict[str, Any] = {}
    predicates: dict[str, Any] = {}

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


def test_dump_invalid_nodes() -> None:
    with pytest.raises(MiddlewareConfigError, match="parallel spec must be a mapping"):
        dump_middleware_config({"parallel": "bad"}, format="json")

    with pytest.raises(MiddlewareConfigError, match="when spec must be a mapping"):
        dump_middleware_config({"when": "bad"}, format="json")

    with pytest.raises(MiddlewareConfigError, match="chain must be a mapping or sequence"):
        dump_middleware_config({"chain": 123}, format="json")

    with pytest.raises(
        MiddlewareConfigError, match="parallel.middleware must be a mapping or sequence"
    ):
        dump_middleware_config({"parallel": {"middleware": 123}}, format="json")

    with pytest.raises(MiddlewareConfigError, match="when.then must be a mapping or sequence"):
        dump_middleware_config({"when": {"then": 123}}, format="json")

    with pytest.raises(MiddlewareConfigError, match="when.else must be a mapping or sequence"):
        dump_middleware_config({"when": {"else": 123}}, format="json")

    with pytest.raises(MiddlewareConfigError, match="Config must be a mapping or sequence"):
        dump_middleware_config(123, format="json")  # type: ignore[arg-type]


def test_dump_parallel_and_when_normalization() -> None:
    config = {
        "parallel": {"middleware": [{"type": "dummy"}]},
        "when": {"then": [{"type": "dummy"}], "else": [{"type": "dummy"}]},
    }
    result = dump_middleware_config(config, format="json")
    assert '"parallel"' in result
    assert '"when"' in result


def test_dump_parallel_without_middleware() -> None:
    result = dump_middleware_config({"parallel": {}}, format="json")
    assert '"parallel"' in result


def test_dump_when_with_none_else() -> None:
    result = dump_middleware_config({"when": {"then": [], "else": None}}, format="json")
    assert '"when"' in result


def test_dump_chain_list_normalization() -> None:
    result = dump_middleware_config([{"chain": [{"type": "dummy"}]}], format="json")
    assert '"chain"' in result


def test_dump_chain_item_type_error() -> None:
    with pytest.raises(MiddlewareConfigError, match="chain items must be mappings"):
        dump_middleware_config({"chain": ["bad"]}, format="json")

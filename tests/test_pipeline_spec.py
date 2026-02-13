from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai_middleware import pipeline_spec as pipeline_spec_module
from pydantic_ai_middleware.base import AgentMiddleware
from pydantic_ai_middleware.builder import MiddlewarePipelineCompiler, MiddlewareRegistry
from pydantic_ai_middleware.config_loaders import load_middleware_config_path
from pydantic_ai_middleware.exceptions import MiddlewareConfigError
from pydantic_ai_middleware.pipeline_spec import (
    PipelineSpec,
    chain_node,
    parallel_node,
    type_node,
    when_node,
)
from pydantic_ai_middleware.strategies import AggregationStrategy


class DummyMiddleware(AgentMiddleware[None]):
    def __init__(self, **config: Any) -> None:
        self.config = config


def test_node_helpers() -> None:
    assert type_node("x") == {"type": "x"}
    assert type_node("x", {"a": 1}) == {"type": "x", "config": {"a": 1}}

    chain = chain_node([{"type": "a"}, {"type": "b"}])
    assert chain == {"chain": [{"type": "a"}, {"type": "b"}]}

    parallel = parallel_node(
        [{"type": "a"}],
        strategy=AggregationStrategy.ALL_MUST_PASS,
        timeout=1.5,
        name="p",
    )
    assert parallel == {
        "parallel": {
            "middleware": [{"type": "a"}],
            "strategy": "all_must_pass",
            "timeout": 1.5,
            "name": "p",
        }
    }

    parallel_string = parallel_node([{"type": "a"}], strategy="race")
    assert parallel_string == {"parallel": {"middleware": [{"type": "a"}], "strategy": "race"}}

    parallel_default = parallel_node([{"type": "a"}], timeout=2.0, name="p2")
    assert parallel_default == {
        "parallel": {"middleware": [{"type": "a"}], "timeout": 2.0, "name": "p2"}
    }

    when = when_node(predicate=True, then=[{"type": "a"}], else_=[{"type": "b"}])
    assert when == {"when": {"predicate": True, "then": [{"type": "a"}], "else": [{"type": "b"}]}}


def test_pipeline_spec_dump_and_compile() -> None:
    reg: MiddlewareRegistry[None] = MiddlewareRegistry(middleware={"dummy": DummyMiddleware})
    compiler = MiddlewarePipelineCompiler(registry=reg)

    spec = (
        PipelineSpec()
        .add_type("dummy", {"id": 1})
        .add_chain([type_node("dummy", {"id": 2})])
        .add_parallel([type_node("dummy", {"id": 3})], strategy="all_must_pass")
        .add_when(predicate=True, then=[type_node("dummy", {"id": 4})])
    )

    dumped = spec.dump(format="json")
    assert '"type": "dummy"' in dumped

    middleware = spec.compile(compiler)
    assert len(middleware) == 4


def test_pipeline_spec_save_and_load_json_path(tmp_path: Path) -> None:
    registry = {"dummy": DummyMiddleware}
    spec = PipelineSpec().add_type("dummy", {"id": 1})
    path = tmp_path / "pipeline.json"

    spec.save(path)
    result = load_middleware_config_path(path, registry=registry)

    assert len(result) == 1
    assert isinstance(result[0], DummyMiddleware)
    assert result[0].config == {"id": 1}


def test_pipeline_spec_dump_yaml_when_available() -> None:
    pytest.importorskip("yaml", reason="PyYAML not installed")
    spec = PipelineSpec().add_type("dummy", {"id": 1})
    result = spec.dump(format="yaml")
    assert "type: dummy" in result


def test_pipeline_spec_dump_unknown_format() -> None:
    spec = PipelineSpec().add_type("dummy")
    with pytest.raises(MiddlewareConfigError, match="Unknown config format"):
        spec.dump(format="toml")


def test_pipeline_spec_save_unknown_extension(tmp_path: Path) -> None:
    spec = PipelineSpec().add_type("dummy")
    with pytest.raises(MiddlewareConfigError, match="Unable to determine config format"):
        spec.save(tmp_path / "pipeline")


def test_pipeline_spec_dump_yaml_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = PipelineSpec().add_type("dummy")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(MiddlewareConfigError, match="YAML support requires PyYAML"):
        spec.dump(format="yaml")


def test_pipeline_spec_detect_format_helpers() -> None:
    assert pipeline_spec_module._detect_format(format="json", path=None) == "json"
    assert pipeline_spec_module._detect_format(format=None, path=Path("pipeline.yaml")) == "yaml"


def test_pipeline_spec_detect_format_unknown_suffix() -> None:
    with pytest.raises(MiddlewareConfigError, match="Unable to determine config format"):
        pipeline_spec_module._detect_format(format=None, path=Path("pipeline.txt"))


def test_pipeline_spec_detect_format_missing_inputs() -> None:
    with pytest.raises(MiddlewareConfigError, match="Unable to determine config format"):
        pipeline_spec_module._detect_format(format=None, path=None)

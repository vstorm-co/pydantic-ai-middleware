# Pipeline Spec

`PipelineSpec` provides a fluent API for constructing middleware pipelines that
can be compiled into middleware instances or exported as configuration files.

## Quick Example

```python
from pydantic_ai_middleware import PipelineSpec
from pydantic_ai_middleware.builder import MiddlewarePipelineCompiler

# Build a pipeline spec with fluent API
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

# Export to YAML file
spec.save("middleware-pipeline.yaml")

# Or compile directly to middleware instances
compiler = MiddlewarePipelineCompiler(registry)
middleware_list = spec.compile(compiler)
```

## Supported Node Types

- **type**: A single middleware by registered name
- **chain**: Sequential execution of multiple nodes
- **parallel**: Concurrent execution with result aggregation
- **when**: Conditional branching based on predicates

## Export as JSON or YAML

```python
from pydantic_ai_middleware.pipeline_spec import PipelineSpec

PipelineSpec().add_type("logging").save("pipeline.json")
PipelineSpec().add_type("logging").save("pipeline.yaml")

# Or get as string
yaml_str = PipelineSpec().add_type("logging").dump("yaml")
json_str = PipelineSpec().add_type("logging").dump("json")
```

## Next Steps

- [Config Loading](config-loading.md) - Load pipelines from config files
- [API Reference](../api/pipeline_spec.md) - PipelineSpec API

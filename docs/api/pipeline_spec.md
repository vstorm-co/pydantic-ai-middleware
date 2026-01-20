# PipelineSpec

Build middleware pipelines in Python with export to JSON/YAML.

`PipelineSpec` provides a fluent API for constructing middleware pipelines that
can be compiled into middleware instances or exported as configuration files.
This is useful for configuration-driven deployments where pipeline definitions
need to be portable.

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

## API Reference

::: pydantic_ai_middleware.pipeline_spec.PipelineSpec
    options:
      show_source: true
      members:
        - nodes
        - add
        - add_type
        - add_chain
        - add_parallel
        - add_when
        - to_config
        - dump
        - save
        - compile

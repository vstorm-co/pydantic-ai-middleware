# Installation

## Requirements

- Python 3.10 or higher
- pydantic-ai-slim >= 1.39

## Install with pip

```bash
pip install pydantic-ai-middleware
```

## Install with uv

```bash
uv add pydantic-ai-middleware
```

## Install from source

```bash
git clone https://github.com/vstorm-co/pydantic-ai-middleware.git
cd pydantic-ai-middleware
pip install -e .
```

## Development Installation

For development, install with all dependencies:

```bash
git clone https://github.com/vstorm-co/pydantic-ai-middleware.git
cd pydantic-ai-middleware
make install
```

This will:

1. Install all dependencies
2. Set up pre-commit hooks
3. Prepare the development environment

## Verify Installation

```python
from pydantic_ai_middleware import MiddlewareAgent, AgentMiddleware

print("Installation successful!")
```

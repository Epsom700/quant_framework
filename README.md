# Quant Framework

> An open, pluggable framework for composable quantitative workflows. Start with FRED. Expand to anything.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the same three-layer contract (immutable evaluator, agent sandbox, human direction), applied to quantitative finance as an extensible framework.

**This is a framework — not a product.** FRED is the hello-world connector. Everything else is an extension of the same pattern.

---

## Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **FRED API Key** — [Get one free from FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd quant_framework

# Install all dependencies
uv sync
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (or export directly):

```bash
# .env
FRED_API_KEY=your_api_key_here
```

### Persona Config

Edit `configs/persona.yaml` to control which functions and connectors your MCP server exposes:

```yaml
name: "Quant Research Agent"
description: "MCP server exposing quantitative research functions"
host: "127.0.0.1"
port: 8000

functions:
  - run_linear
  - run_random_forest
  - run_svr
  - run_xgboost
  - run_bayesian_ridge
  - run_hmm

connectors:
  - fred
```

### Guardrails Config

Edit `configs/guardrails.yaml` to define validation rules for function outputs:

```yaml
defaults:
  max_records: 10000

rules:
  run_linear:
    max_records: 5000
    required_fields: [model, r_squared, coefficients]
    roles:
      analyst:
        redacted_fields: [model]
```

---

## Usage

### CLI — Start the MCP Server

```bash
# Show available commands
uv run quant --help

# Start the MCP server with SSE transport
uv run quant serve --persona configs/persona.yaml

# Use stdio transport instead
uv run quant serve --persona configs/persona.yaml --transport stdio
```

This will:
1. Register all modelling functions from the `FunctionRegistry`
2. Initialise connectors (auto-connects using `$FRED_API_KEY`)
3. Start the MCP server on `127.0.0.1:8000`

### Connect from Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "quant-framework": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### Run the Example Script

```bash
uv run python examples/basic_usage.py
```

This demonstrates:
1. Querying GDP data from FRED
2. Running linear regression via the `FunctionRegistry`
3. Validating the result through the `GuardrailEngine`

---

## Project Structure

```
quant_framework/
├── pyproject.toml                # Dependencies & CLI entry point
├── configs/
│   ├── persona.yaml              # MCP server persona config
│   └── guardrails.yaml           # Validation rules
├── examples/
│   └── basic_usage.py            # End-to-end demo script
└── quant_framework/              # Package root
    ├── cli.py                    # CLI (quant serve)
    ├── core/
    │   ├── function.py           # @register_function, FunctionRegistry, FunctionResult
    │   └── guardrail.py          # GuardrailEngine, GuardrailViolation
    ├── connectors/
    │   ├── connectors.py         # BaseConnector, ConnectorRegistry
    │   └── fred.py               # FREDConnector (with 24h file cache)
    ├── functions/
    │   └── modelling.py          # Registered modelling functions
    └── mcp/
        └── generator.py          # MCPServerGenerator
```

---

## Core Components

### Connectors

| Connector | Registry Name | Description |
|-----------|--------------|-------------|
| `FREDConnector` | `fred` | Federal Reserve Economic Data with 24h file-based cache |

```python
from quant_framework.connectors import FREDConnector

fred = FREDConnector()
fred.connect({"api_key": "your_key"})
df = fred.query("GDP", observation_start="2020-01-01")
```

### Modelling Functions

All functions are registered with `@register_function` and return a `FunctionResult`:

| Function | Registry Name | Model Type | Key Outputs |
|----------|--------------|------------|-------------|
| `run_linear_regression` | `run_linear` | LinearRegression | coefficients, intercept, r² |
| `run_random_forest` | `run_random_forest` | RandomForestRegressor | feature_importances, r² |
| `run_svr` | `run_svr` | SVR | r² |
| `run_xgboost` | `run_xgboost` | XGBRegressor | feature_importances, r² |
| `run_bayesian_ridge` | `run_bayesian_ridge` | BayesianRidge | posterior_std, alpha\_, lambda\_ |
| `run_hmm` | `run_hmm` | GaussianHMM | hidden_states, transition_matrix, AIC, BIC |

```python
from quant_framework.functions.modelling import run_linear_regression

result = run_linear_regression(df, target="GDP", features=["UNRATE", "FEDFUNDS"])
print(result.output["r_squared"])   # 0.12
print(result.trace_id)              # unique trace ID
```

### Guardrail Engine

```python
from quant_framework.core import GuardrailEngine

engine = GuardrailEngine("configs/guardrails.yaml")
engine.validate("run_linear", result.output)           # passes
engine.validate("run_linear", result.output, role="analyst")  # applies role-specific rules
```

- **Hot-reload**: edits to the YAML take effect immediately (checks file mtime)
- **Per-role overrides**: stricter rules for specific roles

### Function Registry

```python
from quant_framework.core import FunctionRegistry

# List all registered functions
FunctionRegistry.list()                        # ['run_linear', 'run_random_forest', ...]
FunctionRegistry.list_by_category("modelling") # filter by category

# Call by name
result = FunctionRegistry.call("run_linear", df=df, target="GDP")
```

---

## Extending the Framework

### Add a Connector

```python
from quant_framework.connectors.connectors import BaseConnector, ConnectorRegistry

@ConnectorRegistry.register("bloomberg")
class BloombergConnector(BaseConnector):
    def connect(self, config): ...
    def query(self, request, **kwargs): ...
    def get_schema(self): ...
    def health_check(self): ...
```

### Add a Function

```python
from quant_framework.core import register_function, FunctionResult

@register_function(name="my_indicator", category="technical")
def my_indicator(df, window=14):
    result = ...  # your logic
    return FunctionResult(output={"value": result}, metrics={"window": window})
```

The function is automatically available in the `FunctionRegistry` and can be exposed as an MCP tool by adding its name to your persona YAML.

---

## Design Principles

- **Connector-first.** Every data source is a `BaseConnector`. Learn one interface, connect anything.
- **Functions as atoms.** Decorated Python functions that auto-register and auto-expose via MCP.
- **Progressive complexity.** Start with FRED. Add what you need, when you need it.
- **Three-layer contract.** Immutable evaluator (guardrails), agent sandbox (function store), human direction (persona configs).

---

## Contributors

Arjun Singh

## License

MIT

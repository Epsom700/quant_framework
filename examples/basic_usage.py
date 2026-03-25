import os
import sys
from pathlib import Path

# Add project root to sys.path for robust imports in examples
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from quant_framework.connectors import FREDConnector
from quant_framework.core import (
    register_function, FunctionRegistry, FunctionResult,
    GuardrailEngine, GuardrailViolation,
)
from quant_framework.functions.modelling import run_linear_regression
import numpy as np
import pandas as pd

# 1. Using the FREDConnector
def demo_fred():
    print("--- FRED Connector Demo ---")
    api_key = os.getenv("FRED_API_KEY")
    
    fred = FREDConnector()
    print(f"Connector Name: {fred.name}")
    
    if api_key:
        print("Connecting to FRED with API key...")
        fred.connect({"api_key": api_key})
        # Fetching real data
        try:
            df = fred.query("GDP", observation_start="2023-01-01")
            print("Successfully fetched GDP data:")
            print(df.head())
        except Exception as e:
            print(f"Error fetching data: {e}")
    else:
        print("Skipping live data fetch: FRED_API_KEY environment variable not set.")
        # Show static schema anyway
        schema = fred.get_schema()
        print(f"Top 5 Series in Schema (Static):")
        for s in schema['series'][:5]:
            print(f"  - {s['id']}: {s['title']}")
    print()

# 2. Using the Function Registry
@register_function(
    name="simple_moving_average",
    category="statistics",
    input_schema={"data": "list", "window": "int"},
    output_schema={"sma": "float"}
)
def calculate_sma(data, window=3):
    avg = sum(data[-window:]) / window
    return FunctionResult(
        output={"sma": avg},
        metrics={"data_points": len(data)}
    )

def demo_functions():
    print("--- Function Registry Demo ---")
    print(f"Registered Functions: {FunctionRegistry.list()}")
    
    data = [10, 20, 30, 40, 50]
    result = FunctionRegistry.call("simple_moving_average", data, window=3)
    
    print(f"SMA Result: {result.output['sma']}")
    print(f"Metrics: {result.metrics}")
    print(f"Trace ID: {result.trace_id}")
    print()

# 3. End-to-end test: FRED → Linear Regression → Guardrail
def test_e2e_fred_to_guardrail():
    print("--- E2E Test: FRED → Linear Regression → Guardrail ---")
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        print("Skipping E2E test: FRED_API_KEY not set.")
        print()
        return

    # Step 1: Create FREDConnector and query GDP + UNRATE data
    fred = FREDConnector()
    fred.connect({"api_key": api_key})

    gdp = fred.query("GDP", observation_start="2000-01-01")
    unrate = fred.query("UNRATE", observation_start="2000-01-01")
    fedfunds = fred.query("FEDFUNDS", observation_start="2000-01-01")

    # Merge into a single DataFrame, forward-fill mismatched frequencies
    df = gdp.join(unrate, how="outer").join(fedfunds, how="outer")
    df = df.ffill().dropna()

    print(f"  Fetched {len(df)} rows with columns: {list(df.columns)}")

    # Step 2: Run linear regression — predict GDP from UNRATE + FEDFUNDS
    result = run_linear_regression(df, target="GDP", features=["UNRATE", "FEDFUNDS"])

    print(f"  Model type : {result.output['model_type']}")
    print(f"  R²         : {result.output['r_squared']:.4f}")
    print(f"  Coefficients: {result.output['coefficients']}")
    print(f"  Intercept   : {result.output['intercept']:.2f}")
    print(f"  Trace ID    : {result.trace_id[:12]}...")

    # Assert r_squared > 0
    assert result.output["r_squared"] > 0, (
        f"Expected r_squared > 0, got {result.output['r_squared']}"
    )
    print("  ✓ Assertion passed: r_squared > 0")

    # Step 3: Validate through GuardrailEngine
    guardrail_path = Path(__file__).resolve().parent.parent / "configs" / "guardrails.yaml"
    engine = GuardrailEngine(guardrail_path)

    try:
        engine.validate("run_linear", result.output)
        print("  ✓ Guardrail validation passed")
    except GuardrailViolation as e:
        print(f"  ✗ Guardrail violation: {e}")

    print()

if __name__ == "__main__":
    demo_fred()
    demo_functions()
    test_e2e_fred_to_guardrail()

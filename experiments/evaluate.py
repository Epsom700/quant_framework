"""
evaluate.py — Fixed evaluation harness for the autoresearch loop.

DO NOT MODIFY. This is the immutable evaluator in the three-layer contract.
The agent runs this after every mutation to strategy.py.
It prints exactly one line: RESULT composite_score=X.XXXX

Composite score formula:
    0.50 × sharpe_ratio
  + 0.30 × directional_accuracy
  + 0.20 × (1 - max_drawdown)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure quant_framework is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_framework.core.guardrail import GuardrailEngine


def load_data_snapshot(snapshot_dir: Path) -> dict:
    """Load cached FRED data from the snapshot directory."""
    data = {}
    for f in snapshot_dir.glob("*.json"):
        with open(f) as fh:
            data[f.stem] = json.load(fh)
    return data


def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from a series of period returns."""
    excess = returns - risk_free_rate
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(12))  # monthly → annual


def compute_directional_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of periods where predicted direction matches actual."""
    if len(predicted) == 0:
        return 0.0
    pred_dir = np.sign(predicted)
    actual_dir = np.sign(actual)
    return float(np.mean(pred_dir == actual_dir))


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown from a cumulative return series."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / np.where(peak == 0, 1, peak)
    return float(np.min(drawdown))


def main():
    t0 = time.time()

    # ── Paths ─────────────────────────────────────────────────────────────────
    experiments_dir = Path(__file__).resolve().parent
    snapshot_dir = experiments_dir / "data_snapshot"
    guardrails_path = experiments_dir.parent / "configs" / "guardrails.yaml"

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_data_snapshot(snapshot_dir)
    if not data:
        print("ERROR: No data in data_snapshot/. Run `uv run python experiments/prepare_snapshot.py` first.", file=sys.stderr)
        sys.exit(1)

    # ── Import and run the strategy ───────────────────────────────────────────
    try:
        from experiments.strategy import run_strategy
    except ImportError as e:
        print(f"ERROR: Could not import strategy: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_strategy(data)
    except Exception as e:
        print(f"ERROR: Strategy execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Validate required outputs ─────────────────────────────────────────────
    required_keys = {"predictions", "actuals", "returns"}
    if not required_keys.issubset(result.keys()):
        missing = required_keys - result.keys()
        print(f"ERROR: Strategy must return {missing}", file=sys.stderr)
        sys.exit(1)

    predictions = np.array(result["predictions"])
    actuals = np.array(result["actuals"])
    returns = np.array(result["returns"])

    if len(predictions) != len(actuals) or len(predictions) != len(returns):
        print("ERROR: predictions, actuals, and returns must have equal length", file=sys.stderr)
        sys.exit(1)

    # ── Guardrail validation (if strategy returned model outputs) ─────────────
    if guardrails_path.exists() and "model_outputs" in result:
        engine = GuardrailEngine(str(guardrails_path))
        for func_name, output in result["model_outputs"].items():
            try:
                engine.validate(func_name, output)
            except Exception as e:
                print(f"ERROR: Guardrail violation on {func_name}: {e}", file=sys.stderr)
                sys.exit(1)

    # ── Compute composite score ───────────────────────────────────────────────
    sharpe = compute_sharpe(returns)
    dir_acc = compute_directional_accuracy(predictions, actuals)
    cumulative = np.cumprod(1 + returns)
    max_dd = compute_max_drawdown(cumulative)

    composite_score = (
        0.50 * sharpe
        + 0.30 * dir_acc
        + 0.20 * (1 - abs(max_dd))
    )

    elapsed = time.time() - t0

    # ── Output ────────────────────────────────────────────────────────────────
    # The agent parses this line. Do not change the format.
    print(f"RESULT composite_score={composite_score:.4f}")

    # Additional context for the agent (non-parsed, informational)
    print(f"  sharpe={sharpe:.4f}  dir_acc={dir_acc:.4f}  max_dd={max_dd:.4f}  elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()

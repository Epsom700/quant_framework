"""
strategy.py — The agent's sandbox.

This is the ONLY file the autoresearch agent is allowed to modify.
It defines: data selection, feature engineering, model choice, and signal logic.

Must expose a single function:
    run_strategy(data: dict) -> dict

The returned dict must contain:
    predictions: list[float]  — predicted values/signals per period
    actuals:     list[float]  — actual target values per period
    returns:     list[float]  — strategy returns per period

Optionally:
    model_outputs: dict[str, dict]  — raw FunctionResult outputs keyed by function name
                                      (passed to GuardrailEngine for validation)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_framework.core.function import FunctionRegistry


# ── CONFIG ────────────────────────────────────────────────────────────────────
# The agent mutates these values and the logic below.

TARGET_SERIES = "GDP"
FEATURE_SERIES = ["UNRATE", "FEDFUNDS", "GS10", "CPIAUCSL"]
MODEL_NAME = "run_linear"
TRAIN_RATIO = 0.7
LOOKBACK_PERIODS = 4  # quarters of lag features


def run_strategy(data: dict) -> dict:
    """
    Baseline strategy: linear regression on FRED macro series.
    The agent will evolve this function across research cycles.
    """

    # ── 1. Build DataFrame from snapshot ──────────────────────────────────────
    frames = {}
    for series_name in [TARGET_SERIES] + FEATURE_SERIES:
        if series_name not in data:
            raise ValueError(f"Series '{series_name}' not found in data snapshot")
        records = data[series_name]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={"value": series_name})
        frames[series_name] = df[[series_name]]

    combined = pd.concat(frames.values(), axis=1, join="inner").dropna()

    if len(combined) < 20:
        raise ValueError(f"Not enough overlapping data: {len(combined)} rows")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    features_df = combined[FEATURE_SERIES].copy()

    # Add lagged features
    for lag in range(1, LOOKBACK_PERIODS + 1):
        for col in FEATURE_SERIES:
            features_df[f"{col}_lag{lag}"] = combined[col].shift(lag)

    features_df = features_df.dropna()
    target = combined[TARGET_SERIES].loc[features_df.index]

    # Align
    common_idx = features_df.index.intersection(target.index)
    features_df = features_df.loc[common_idx]
    target = target.loc[common_idx]

    # ── 3. Train/test split (temporal, no leakage) ────────────────────────────
    split_idx = int(len(features_df) * TRAIN_RATIO)
    X_train = features_df.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]

    if len(X_test) < 5:
        raise ValueError(f"Test set too small: {len(X_test)} rows")

    # ── 4. Model training via FunctionRegistry ────────────────────────────────
    train_df = X_train.copy()
    train_df[TARGET_SERIES] = y_train

    result = FunctionRegistry.call(
        MODEL_NAME,
        df=train_df,
        target=TARGET_SERIES,
        features=list(X_train.columns),
    )

    model_outputs = {MODEL_NAME: result.output}

    # ── 5. Generate predictions ───────────────────────────────────────────────
    # For the baseline, use the model's coefficients for manual prediction.
    # More sophisticated strategies can call predict methods or ensemble models.
    if MODEL_NAME == "run_linear" and "coefficients" in result.output:
        coeffs = np.array(result.output["coefficients"])
        intercept = result.output.get("intercept", 0.0)
        predictions = X_test.values @ coeffs + intercept
    else:
        # Fallback: naive prediction (last known value)
        predictions = np.full(len(X_test), y_train.iloc[-1])

    actuals = y_test.values

    # ── 6. Compute strategy returns ───────────────────────────────────────────
    # Simple directional strategy: go long if predicted change is positive
    predicted_direction = np.sign(np.diff(predictions, prepend=predictions[0]))
    actual_returns = np.diff(actuals, prepend=actuals[0]) / np.where(
        np.abs(actuals) < 1e-10, 1.0, np.abs(actuals)
    )
    strategy_returns = predicted_direction * actual_returns

    return {
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        "returns": strategy_returns.tolist(),
        "model_outputs": model_outputs,
    }

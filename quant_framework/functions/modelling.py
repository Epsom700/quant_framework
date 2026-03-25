"""Modelling functions registered with the quant_framework FunctionRegistry."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from hmmlearn.hmm import GaussianHMM

from quant_framework.core.function import FunctionResult, register_function


# ── Helper ───────────────────────────────────────────────────────────────────

def _prepare_data(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    y = df[target]
    X = df[features] if features else df.drop(columns=[target])
    return X, y


def _build_result(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    elapsed: float,
    model_name: str,
) -> FunctionResult:
    """Create a standardised FunctionResult from a fitted sklearn-style model."""
    predictions = model.predict(X)
    residuals = y.values - predictions

    r_squared = float(model.score(X, y))

    output: Dict[str, Any] = {
        "model": model,
        "model_type": model_name,
        "r_squared": r_squared,
        "predictions": predictions.tolist(),
        "residuals": residuals.tolist(),
    }

    # Linear models expose coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_
        output["coefficients"] = (
            coefs.tolist() if isinstance(coefs, np.ndarray) else [float(coefs)]
        )
    if hasattr(model, "intercept_"):
        intercept = model.intercept_
        if isinstance(intercept, np.ndarray):
            output["intercept"] = intercept.tolist()
        else:
            output["intercept"] = float(intercept)

    # Tree / ensemble models expose feature importances
    if hasattr(model, "feature_importances_"):
        output["feature_importances"] = dict(
            zip(X.columns.tolist(), model.feature_importances_.tolist())
        )

    metrics: Dict[str, Any] = {
        "n_samples": len(y),
        "n_features": X.shape[1],
        "training_time_s": round(elapsed, 4),
    }

    return FunctionResult(output=output, metrics=metrics)


# ── Registered modelling functions ───────────────────────────────────────────

@register_function(
    name="run_linear",
    category="modelling",
    input_schema={
        "df": "pd.DataFrame",
        "target": "str",
        "features": "Optional[List[str]]",
    },
    output_schema={
        "model": "LinearRegression",
        "r_squared": "float",
        "coefficients": "List[float]",
        "intercept": "float",
        "residuals": "List[float]",
    },
)
def run_linear_regression(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
) -> FunctionResult:
    """Fit a Linear Regression and return results with coefficients and residuals."""
    X, y = _prepare_data(df, target, features)
    model = LinearRegression()
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    return _build_result(model, X, y, elapsed, "LinearRegression")


@register_function(
    name="run_random_forest",
    category="modelling",
    input_schema={
        "df": "pd.DataFrame",
        "target": "str",
        "features": "Optional[List[str]]",
        "n_estimators": "int (default 100)",
        "max_depth": "Optional[int]",
    },
    output_schema={
        "model": "RandomForestRegressor",
        "r_squared": "float",
        "feature_importances": "Dict[str, float]",
        "residuals": "List[float]",
    },
)
def run_random_forest(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> FunctionResult:
    """Fit a Random Forest Regressor and return results with feature importances."""
    X, y = _prepare_data(df, target, features)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    return _build_result(model, X, y, elapsed, "RandomForestRegressor")


@register_function(
    name="run_svr",
    category="modelling",
    input_schema={
        "df": "pd.DataFrame",
        "target": "str",
        "features": "Optional[List[str]]",
        "kernel": "str (default 'rbf')",
        "C": "float (default 1.0)",
        "epsilon": "float (default 0.1)",
    },
    output_schema={
        "model": "SVR",
        "r_squared": "float",
        "residuals": "List[float]",
    },
)
def run_svr(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    kernel: str = "rbf",
    C: float = 1.0,
    epsilon: float = 0.1,
) -> FunctionResult:
    """Fit a Support Vector Regressor and return results."""
    X, y = _prepare_data(df, target, features)
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    return _build_result(model, X, y, elapsed, "SVR")


@register_function(
    name="run_xgboost",
    category="modelling",
    input_schema={
        "df": "pd.DataFrame",
        "target": "str",
        "features": "Optional[List[str]]",
        "n_estimators": "int (default 100)",
        "max_depth": "int (default 6)",
        "learning_rate": "float (default 0.1)",
    },
    output_schema={
        "model": "XGBRegressor",
        "r_squared": "float",
        "feature_importances": "Dict[str, float]",
        "residuals": "List[float]",
    },
)
def run_xgboost(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> FunctionResult:
    """Fit an XGBoost Regressor and return results with feature importances."""
    X, y = _prepare_data(df, target, features)
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=0,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    return _build_result(model, X, y, elapsed, "XGBRegressor")


# ── Bayesian Models ──────────────────────────────────────────────────────────

@register_function(
    name="run_bayesian_ridge",
    category="modelling",
    input_schema={
        "df": "pd.DataFrame",
        "target": "str",
        "features": "Optional[List[str]]",
        "max_iter": "int (default 300)",
        "alpha_init": "Optional[float]",
        "lambda_init": "Optional[float]",
    },
    output_schema={
        "model": "BayesianRidge",
        "r_squared": "float",
        "coefficients": "List[float]",
        "intercept": "float",
        "residuals": "List[float]",
        "posterior_std": "List[float]",
        "alpha_": "float",
        "lambda_": "float",
    },
)
def run_bayesian_ridge(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    max_iter: int = 300,
    alpha_init: Optional[float] = None,
    lambda_init: Optional[float] = None,
) -> FunctionResult:
    """Fit a Bayesian Ridge Regression.

    Returns standard regression outputs plus Bayesian-specific posterior
    standard deviations and estimated precision parameters (alpha, lambda).
    """
    X, y = _prepare_data(df, target, features)

    kwargs: Dict[str, Any] = {"max_iter": max_iter}
    if alpha_init is not None:
        kwargs["alpha_init"] = alpha_init
    if lambda_init is not None:
        kwargs["lambda_init"] = lambda_init

    model = BayesianRidge(**kwargs)
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    result = _build_result(model, X, y, elapsed, "BayesianRidge")

    # Bayesian-specific: posterior std of predictions & precision params
    _, y_std = model.predict(X, return_std=True)
    result.output["posterior_std"] = y_std.tolist()
    result.output["alpha_"] = float(model.alpha_)
    result.output["lambda_"] = float(model.lambda_)

    return result


# ── Hidden Markov Models ─────────────────────────────────────────────────────

@register_function(
    name="run_hmm",
    category="modelling",
    input_schema={
        "df": "pd.DataFrame",
        "features": "Optional[List[str]]",
        "n_states": "int (default 3)",
        "covariance_type": "str (default 'full')",
        "n_iter": "int (default 100)",
    },
    output_schema={
        "model": "GaussianHMM",
        "hidden_states": "List[int]",
        "transition_matrix": "List[List[float]]",
        "state_means": "List[List[float]]",
        "state_covariances": "List[List[List[float]]]",
        "log_likelihood": "float",
        "aic": "float",
        "bic": "float",
    },
)
def run_hmm(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 100,
    random_state: int = 42,
) -> FunctionResult:
    """Fit a Gaussian Hidden Markov Model.

    Unlike the regression functions, HMMs are unsupervised — there is no
    target column.  The function discovers *n_states* hidden regimes in the
    data and returns the decoded state sequence, transition matrix, and
    per-state Gaussian parameters.
    """
    X = df[features].values if features else df.values
    n_samples, n_features = X.shape

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )

    t0 = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - t0

    hidden_states = model.predict(X)
    log_likelihood = float(model.score(X))

    # Compute AIC / BIC
    n_params = (
        n_states * n_features          # means
        + n_states * n_features         # covariances (diagonal lower bound)
        + n_states * n_states           # transition matrix
        + n_states                      # start probabilities
    )
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    output: Dict[str, Any] = {
        "model": model,
        "model_type": "GaussianHMM",
        "hidden_states": hidden_states.tolist(),
        "transition_matrix": model.transmat_.tolist(),
        "state_means": model.means_.tolist(),
        "state_covariances": model.covars_.tolist(),
        "log_likelihood": log_likelihood,
        "aic": float(aic),
        "bic": float(bic),
    }

    metrics: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_states": n_states,
        "training_time_s": round(elapsed, 4),
    }

    return FunctionResult(output=output, metrics=metrics)

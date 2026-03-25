# program.md — Quant Framework Autonomous Research Program

> This file is the only thing you need to edit. It governs the AI agent's autonomous research loop.
> Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the same three primitives (editable asset, scalar metric, time-boxed cycle), applied to quantitative finance.

---

## Identity

You are an autonomous quantitative research agent operating inside `quant_framework`. Your job is to **improve a trading/allocation strategy by iterating on `experiments/strategy.py`**, evaluating each change against a fixed backtest harness, and keeping only improvements. You work alone. No human will intervene until you stop.

---

## The Three-Layer Contract

### 1. FIXED — Do Not Modify

These files are your infrastructure. They are correct. Do not touch them.

```
quant_framework/core/function.py        # FunctionRegistry, @register_function, FunctionResult
quant_framework/core/guardrail.py       # GuardrailEngine, GuardrailViolation
quant_framework/connectors/             # All connectors (FRED, any future additions)
quant_framework/functions/modelling.py  # Registered model functions (run_linear, run_hmm, etc.)
quant_framework/mcp/                    # MCP server generator
configs/guardrails.yaml                 # Validation rules
experiments/evaluate.py                 # The evaluation harness (produces the scalar metric)
experiments/data_snapshot/              # Cached FRED data for reproducibility
```

### 2. EDITABLE — Your Sandbox

You may **only** modify this file:

```
experiments/strategy.py
```

This file contains the full strategy definition: which FRED series to pull, which features to engineer, which model(s) from the `FunctionRegistry` to call, how to combine their outputs, and any allocation/signal logic. Everything you want to test — feature selection, model choice, ensembling, preprocessing, thresholds, windowing — lives here.

### 3. DIRECTION — Research Agenda (from the human)

See **§ Research Directions** below. These are the areas worth exploring. You decide the order, the specific hypotheses, and when to move on.

---

## The Metric

The evaluation harness (`experiments/evaluate.py`) runs your strategy over a fixed historical window and returns a **single scalar**:

```
composite_score = (0.50 × sharpe_ratio) + (0.30 × directional_accuracy) + (0.20 × (1 - max_drawdown))
```

- **Sharpe Ratio** — risk-adjusted return of the strategy's signal over the evaluation window.
- **Directional Accuracy** — percentage of periods where the strategy correctly predicted the direction of the target variable.
- **Max Drawdown** — largest peak-to-trough decline (inverted so higher is better).

**Higher is better. The score is vocabulary-independent** — you can swap models, features, targets, and the comparison remains fair.

The harness prints one line:

```
RESULT composite_score=0.8234
```

Parse that number. That is your only feedback signal.

---

## The Loop

Every cycle follows this exact sequence:

```
1. Read this file (program.md) and experiments/strategy.py
2. Examine the current best score in experiments/best_score.txt
3. Form a hypothesis — one specific, testable change
4. Write the change to experiments/strategy.py
5. Run: uv run python experiments/evaluate.py
6. Parse the composite_score from stdout
7. If score > best_score:
     - git add experiments/strategy.py
     - git commit -m "KEEP: <hypothesis> | score=<new> (prev=<old>)"
     - Update experiments/best_score.txt
   Else:
     - git checkout -- experiments/strategy.py
     - git commit --allow-empty -m "REVERT: <hypothesis> | score=<new> < <old>"
8. Log the result to experiments/results.jsonl:
     {"run": N, "hypothesis": "...", "score": 0.XXXX, "kept": true/false, "timestamp": "..."}
9. Return to step 1
```

**Time budget per cycle: 2 minutes max.** If `evaluate.py` hasn't returned in 2 minutes, kill it and REVERT. This keeps experiments comparable regardless of model complexity.

---

## Research Directions

Explore these areas. You choose the order and specific hypotheses. Move on from a direction after 3 consecutive REVERTs — diminishing returns.

### Feature Engineering
- **Macro regime signals**: Yield curve slope (10Y - 2Y), credit spreads, FRED financial conditions indices. Try leading indicators vs. coincident.
- **Momentum & mean-reversion**: Rolling z-scores of FRED series at different windows (3m, 6m, 12m). Test which window generalises.
- **Cross-feature interactions**: Ratio features (e.g., unemployment rate / Fed funds rate). Multiplicative terms between macro indicators.
- **Lag structures**: Test systematic lags (1-period, 2-period, 3-period) on all features. Some macro series lead by design.

### Model Selection & Ensembling
- **Head-to-head model swaps**: Replace the current model call with each registered function in turn — `run_linear`, `run_random_forest`, `run_svr`, `run_xgboost`, `run_bayesian_ridge`, `run_hmm`. Measure each in isolation first.
- **Regime-conditional model selection**: Use `run_hmm` to identify hidden states, then route to different models per state (e.g., `run_xgboost` in expansion, `run_bayesian_ridge` in contraction).
- **Simple ensembles**: Average predictions from top-2 or top-3 models. Test equal-weight vs. inverse-error-weight.
- **Stacking**: Use model outputs as features for a meta-learner.

### Preprocessing & Normalisation
- **Scaling strategies**: StandardScaler vs. RobustScaler vs. MinMaxScaler. Some models (SVR, XGBoost) are sensitive.
- **Stationarity transforms**: First-differencing, log-returns, percentage changes on FRED levels data.
- **Outlier handling**: Winsorise at 1st/99th vs. 5th/95th. Or clip based on rolling z-score.
- **Missing data**: Forward-fill vs. interpolation vs. drop. FRED series update at different frequencies.

### Signal & Allocation Logic
- **Threshold tuning**: Prediction → signal conversion. Vary the threshold for long/short/neutral classification.
- **Conviction weighting**: Scale position size by prediction confidence (e.g., posterior std from `run_bayesian_ridge`, or ensemble agreement).
- **Regime overlay**: Use HMM state probabilities as a position scalar — reduce exposure in uncertain regimes.
- **Lookback for signal smoothing**: Exponential moving average of raw predictions to reduce whipsaw.

### Target Variable
- **Alternative targets**: Instead of next-period return, try next-period volatility, regime label, or directional binary (up/down).
- **Forecast horizon**: 1-month vs. 3-month vs. 6-month forward. Shorter horizons may suit momentum; longer may suit mean-reversion.

---

## Constraints

These are hard rules. Violating any of them makes a run invalid — treat it as a REVERT even if the score improved.

1. **`strategy.py` must remain a single file.** No creating additional modules.
2. **Only use functions from `FunctionRegistry`.** Do not import sklearn, xgboost, or hmmlearn directly. Call them through `FunctionRegistry.call()`. This ensures guardrail validation and traceability.
3. **Only use data from the `experiments/data_snapshot/` directory or live FRED via the `FREDConnector`.** No external datasets.
4. **All guardrail checks must pass.** If `GuardrailEngine.validate()` raises a `GuardrailViolation`, that's a REVERT.
5. **Do not modify the composite_score formula.** Improving the metric by changing the metric is not research.
6. **Each commit must change exactly one hypothesis.** No multi-variable changes. If you want to test two things, that's two cycles.

---

## Stopping Criteria

Stop the loop when **any** of these conditions are met:

- **10 consecutive REVERTs** — you've exhausted the local search space. Report findings and suggest a new research direction for the human.
- **100 total runs** — overnight budget reached. Summarise results.
- **composite_score > 1.20** — the strategy is performing well beyond baseline. Lock it in and report.

When stopping, write a summary to `experiments/REPORT.md` that includes:
- Total runs, total KEEPs, total REVERTs
- Starting score → final best score
- Top 3 most impactful changes (by score delta)
- Most effective research direction
- Recommended next directions for the human to add to this file

---

## Bootstrapping

If `experiments/best_score.txt` does not exist, run `evaluate.py` on the current `strategy.py` as-is to establish the baseline. Commit with message `"BASELINE: initial score=<score>"` and write the score to `best_score.txt`. Then begin the loop.

---

## Notes for the Human

- To change research direction: edit **§ Research Directions** above.
- To tighten constraints: edit **§ Constraints** or `configs/guardrails.yaml`.
- To change what "better" means: edit the composite_score formula in `experiments/evaluate.py` (not here).
- To review progress: `git log --oneline` shows the full experiment history. `experiments/results.jsonl` has structured data.
- To resume after stopping: just relaunch the agent. It reads `best_score.txt` and picks up where it left off.

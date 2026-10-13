import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcadgreeks.adaptive.validity import admissible_methods
ESTIMATES = pd.read_csv(ROOT / "results" / "adaptive_estimator_benchmark.csv")
SUMMARY = pd.read_csv(ROOT / "results" / "adaptive_estimator_summary.csv")
PAYOFFS = {"EuropeanCall", "EuropeanPut", "CashDigitalCall"}
GREEKS = {"Delta", "Gamma", "Vega", "Rho"}
METHODS = {"AdaptiveSelected", "Pathwise", "LRM", "FD_CRN"}
NUMERIC = [
    "estimate",
    "reference",
    "error",
    "absolute_error",
    "standard_error",
    "selection_seconds",
    "production_warmup_seconds",
    "production_seconds",
    "total_elapsed_seconds",
]

assert len(ESTIMATES) == 480
assert len(SUMMARY) == 48
assert set(ESTIMATES["payoff"]) == PAYOFFS
assert set(ESTIMATES["greek"]) == GREEKS
assert set(ESTIMATES["method"]) == METHODS
assert np.isfinite(ESTIMATES[NUMERIC].to_numpy()).all()
assert (ESTIMATES["absolute_error"] >= 0).all()
assert (ESTIMATES["standard_error"] >= 0).all()
assert (ESTIMATES["selection_seconds"] >= 0).all()
assert (ESTIMATES["production_warmup_seconds"] >= 0).all()
assert (ESTIMATES["production_seconds"] > 0).all()
assert (ESTIMATES["total_elapsed_seconds"] + 1e-6 >= ESTIMATES["selection_seconds"] + ESTIMATES["production_warmup_seconds"] + ESTIMATES["production_seconds"]).all()
assert (ESTIMATES["paths"] == 200000).all()
assert (SUMMARY["replicate_count"] == 10).all()
assert np.isfinite(SUMMARY[["bias", "rmse", "mean_selection_seconds", "mean_production_warmup_seconds", "mean_production_seconds", "mean_total_elapsed_seconds"]].to_numpy()).all()

for (payoff, greek, method), group in ESTIMATES.groupby(["payoff", "greek", "method"]):
    assert group["replicate"].nunique() == 10
    if method == "AdaptiveSelected":
        allowed = set(admissible_methods(payoff, greek))
        assert set(group["selected_method"]).issubset(allowed)
    else:
        assert group["selected_method"].notna().all()

assert not ESTIMATES.loc[
    (ESTIMATES["payoff"] == "CashDigitalCall") &
    (ESTIMATES["method"] == "AdaptiveSelected"),
    "selected_method",
].eq("Pathwise").any()

environment_path = ROOT / "results" / "adaptive_benchmark_environment.txt"
environment_text = environment_path.read_text(encoding="utf-8")
environment_text = environment_text.replace(
    "VALIDATION: run experiments/validate_benchmark.py after this script",
    "VALIDATION: passed",
)
environment_path.write_text(environment_text, encoding="utf-8")
print("Adaptive benchmark validation passed: 480 estimate rows, 48 summary rows, 10 replications per method-payoff-Greek group.")

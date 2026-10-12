import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.validity import admissible_methods, select_valid_candidate
from mcadgreeks.estimators.digital_estimators import pathwise_ad_samples, lrm_samples, fd_greek_samples
from mcadgreeks.references.digital import delta_cash_digital_call, vega_cash_digital_call, rho_cash_digital_call

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
PARAMETERS = (S0, K, R, SIGMA, T)
GREEKS = ("Delta", "Vega", "Rho")
METHODS = ("Pathwise", "LRM", "FD_CRN")
REPLICATES = 12
BATCH_PATHS = 8192
BUDGET_SECONDS = 0.15
PILOT_REPEATS = 3
SEED = 20261012
STEPS = {"Delta": 0.1, "Vega": 0.005, "Rho": 0.01}
REFERENCES = {
    "Delta": delta_cash_digital_call(*PARAMETERS),
    "Vega": vega_cash_digital_call(*PARAMETERS),
    "Rho": rho_cash_digital_call(*PARAMETERS),
}

def make_functions(greek):
    return {
        "Pathwise": jax.jit(lambda z: pathwise_ad_samples(z, greek, *PARAMETERS)),
        "LRM": jax.jit(lambda z: lrm_samples(z, greek, *PARAMETERS)),
        "FD_CRN": jax.jit(lambda z: fd_greek_samples(z, greek, *PARAMETERS, STEPS[greek])),
    }

FUNCTIONS = {greek: make_functions(greek) for greek in GREEKS}
key = jax.random.key(SEED)

for greek in GREEKS:
    warm_z = jnp.zeros((BATCH_PATHS,), dtype=jnp.float64)
    for method in METHODS:
        FUNCTIONS[greek][method](warm_z).block_until_ready()

def draw_batch(key, fn):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (BATCH_PATHS,), dtype=jnp.float64)
    values = fn(z)
    values.block_until_ready()
    return key, np.asarray(values, dtype=np.float64)

def score_candidates(key, greek):
    key, pilot_key = jax.random.split(key)
    pilot_z = jax.random.normal(pilot_key, (BATCH_PATHS,), dtype=jnp.float64)
    pilot_z.block_until_ready()
    candidates = []

    for method in METHODS:
        fn = FUNCTIONS[greek][method]
        values = fn(pilot_z)
        values.block_until_ready()
        samples = np.asarray(values, dtype=np.float64)
        durations = []

        for _ in range(PILOT_REPEATS):
            start = time.perf_counter()
            output = fn(pilot_z)
            output.block_until_ready()
            samples = np.asarray(output, dtype=np.float64)
            durations.append(time.perf_counter() - start)

        variance = float(np.var(samples, ddof=1))
        seconds_per_path = float(np.median(durations)) / BATCH_PATHS

        candidates.append({
            "method": method,
            "pilot_variance": variance,
            "seconds_per_path": seconds_per_path,
            "variance_time_score": variance * seconds_per_path,
        })

    return key, candidates

def summarize_result(greek, method, selected, total, total_sq, n, elapsed, pilot_seconds, replicate):
    estimate = total / n
    variance = max(0.0, (total_sq - total * total / n) / max(1, n - 1))
    standard_error = np.sqrt(variance / n)
    reference = REFERENCES[greek]

    return {
        "replicate": replicate,
        "greek": greek,
        "method": method,
        "selected_estimator": selected,
        "estimate": estimate,
        "reference": reference,
        "error": estimate - reference,
        "absolute_error": abs(estimate - reference),
        "standard_error": standard_error,
        "covered_95": abs(estimate - reference) <= 1.96 * standard_error,
        "production_paths": n,
        "budget_seconds": BUDGET_SECONDS,
        "elapsed_seconds": elapsed,
        "pilot_seconds": pilot_seconds,
    }

def run_fixed(key, greek, method, replicate):
    start = time.perf_counter()
    total = 0.0
    total_sq = 0.0
    n = 0
    fn = FUNCTIONS[greek][method]

    while n == 0 or time.perf_counter() - start < BUDGET_SECONDS:
        key, values = draw_batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += values.size

    elapsed = time.perf_counter() - start
    result = summarize_result(
        greek, "Fixed_" + method, method, total, total_sq, n,
        elapsed, 0.0, replicate
    )
    return key, result

def run_adaptive(key, greek, constrained, replicate):
    start = time.perf_counter()
    key, candidates = score_candidates(key, greek)

    if constrained:
        selected = select_valid_candidate("CashDigitalCall", greek, candidates)["method"]
        policy = "Adaptive_Validity"
    else:
        selected = min(candidates, key=lambda item: item["variance_time_score"])["method"]
        policy = "Adaptive_Unconstrained"

    pilot_seconds = time.perf_counter() - start
    fn = FUNCTIONS[greek][selected]
    total = 0.0
    total_sq = 0.0
    n = 0

    while n == 0 or time.perf_counter() - start < BUDGET_SECONDS:
        key, values = draw_batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += values.size

    elapsed = time.perf_counter() - start
    result = summarize_result(
        greek, policy, selected, total, total_sq, n,
        elapsed, pilot_seconds, replicate
    )
    return key, result

records = []

for replicate in range(REPLICATES):
    for greek in GREEKS:
        for method in METHODS:
            key, result = run_fixed(key, greek, method, replicate)
            records.append(result)

        key, result = run_adaptive(key, greek, False, replicate)
        records.append(result)

        key, result = run_adaptive(key, greek, True, replicate)
        records.append(result)

    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

df = pd.DataFrame(records)
df.to_csv("results/validity_regret_estimates.csv", index=False)

summary = df.groupby(["greek", "method"], as_index=False).agg(
    mean_estimate=("estimate", "mean"),
    bias=("error", "mean"),
    rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    mean_absolute_error=("absolute_error", "mean"),
    mean_standard_error=("standard_error", "mean"),
    empirical_coverage_95=("covered_95", "mean"),
    mean_production_paths=("production_paths", "mean"),
    mean_elapsed_seconds=("elapsed_seconds", "mean"),
    mean_pilot_seconds=("pilot_seconds", "mean"),
    replicates=("replicate", "nunique"),
)
summary.to_csv("results/validity_regret_summary.csv", index=False)

regret_rows = []

for greek in GREEKS:
    subset = summary[summary["greek"] == greek]
    fixed = subset[subset["method"].isin(["Fixed_LRM", "Fixed_FD_CRN"])]
    oracle = fixed.loc[fixed["rmse"].idxmin()]
    adaptive = subset[subset["method"].isin(["Adaptive_Unconstrained", "Adaptive_Validity"])]

    for _, row in adaptive.iterrows():
        regret_rows.append({
            "greek": greek,
            "policy": row["method"],
            "oracle_fixed_method": oracle["method"],
            "oracle_fixed_rmse": oracle["rmse"],
            "policy_rmse": row["rmse"],
            "empirical_rmse_gap": row["rmse"] - oracle["rmse"],
            "mean_pilot_seconds": row["mean_pilot_seconds"],
            "mean_elapsed_seconds": row["mean_elapsed_seconds"],
        })

regret = pd.DataFrame(regret_rows)
regret.to_csv("results/validity_regret_comparison.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, greek in zip(axes, GREEKS):
    subset = summary[summary["greek"] == greek]
    ax.bar(subset["method"], subset["rmse"])
    ax.set_title(greek)
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("figures/validity_regret_comparison.png", dpi=300)
plt.close(fig)

print("\nBENCHMARK SUMMARY")
print(summary.to_string(index=False))
print("\nEMPIRICAL RMSE GAP TO BEST VALID FIXED BASELINE")
print(regret.to_string(index=False))

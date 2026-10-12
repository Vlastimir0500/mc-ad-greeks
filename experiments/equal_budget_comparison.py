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

from mcadgreeks.estimators.greek_estimators import estimator_samples
from mcadgreeks.references.black_scholes import delta_call, vega_call, rho_call

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
PARAMETERS = (S0, K, R, SIGMA, T)
GREEKS = ("Delta", "Vega", "Rho")
METHODS = ("Pathwise", "LRM")
BATCH_PATHS = 8192
BUDGET_SECONDS = 0.25
REPLICATES = 8
SEED = 20261012
REFERENCES = {
    "Delta": delta_call(S0, K, R, SIGMA, T),
    "Vega": vega_call(S0, K, R, SIGMA, T),
    "Rho": rho_call(S0, K, R, SIGMA, T),
}

functions = {}
key = jax.random.key(SEED)

for greek in GREEKS:
    for method in METHODS:
        fn = jax.jit(
            lambda z, g=greek, m=method:
                estimator_samples(z, g, m, *PARAMETERS)
        )
        key, warm_key = jax.random.split(key)
        warm_z = jax.random.normal(warm_key, (BATCH_PATHS,), dtype=jnp.float64)
        fn(warm_z).block_until_ready()
        functions[(greek, method)] = fn

def batch(key, fn):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (BATCH_PATHS,), dtype=jnp.float64)
    values = fn(z)
    values.block_until_ready()
    return key, np.asarray(values, dtype=np.float64)

def make_result(greek, method, selected, total, total_sq, n, elapsed, pilot, replicate):
    if n == 0:
        raise RuntimeError("The time budget produced no production samples")
    estimate = total / n
    variance = max(0.0, (total_sq - total * total / n) / max(1, n - 1))
    reference = REFERENCES[greek]
    return {
        "replicate": replicate,
        "greek": greek,
        "method": method,
        "selected_method": selected,
        "estimate": estimate,
        "reference": reference,
        "error": estimate - reference,
        "absolute_error": abs(estimate - reference),
        "standard_error": np.sqrt(variance / n),
        "production_paths": n,
        "budget_seconds": BUDGET_SECONDS,
        "elapsed_seconds": elapsed,
        "pilot_seconds": pilot,
        "timing_scope": "warmed_jit_wall_clock",
    }

def run_fixed(key, greek, method, replicate):
    fn = functions[(greek, method)]
    total = 0.0
    total_sq = 0.0
    n = 0
    start = time.perf_counter()

    while time.perf_counter() - start < BUDGET_SECONDS:
        key, values = batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += values.size

    elapsed = time.perf_counter() - start
    return key, make_result(
        greek, method, method, total, total_sq, n, elapsed, 0.0, replicate
    )

def run_adaptive(key, greek, replicate):
    start = time.perf_counter()
    key, pilot_key = jax.random.split(key)
    pilot_z = jax.random.normal(
        pilot_key, (BATCH_PATHS,), dtype=jnp.float64
    )
    pilot_z.block_until_ready()

    candidates = []

    for method in METHODS:
        fn = functions[(greek, method)]
        durations = []
        samples = None

        for _ in range(3):
            tick = time.perf_counter()
            values = fn(pilot_z)
            values.block_until_ready()
            samples = np.asarray(values, dtype=np.float64)
            durations.append(time.perf_counter() - tick)

        variance = float(np.var(samples, ddof=1))
        seconds_per_path = float(np.median(durations)) / BATCH_PATHS
        candidates.append({
            "method": method,
            "variance": variance,
            "seconds_per_path": seconds_per_path,
            "score": variance * seconds_per_path,
        })

    selected = min(candidates, key=lambda item: item["score"])["method"]
    pilot_elapsed = time.perf_counter() - start
    fn = functions[(greek, selected)]

    total = 0.0
    total_sq = 0.0
    n = 0

    while time.perf_counter() - start < BUDGET_SECONDS:
        key, values = batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += values.size

    elapsed = time.perf_counter() - start
    return key, make_result(
        greek, "Adaptive", selected, total, total_sq, n,
        elapsed, pilot_elapsed, replicate
    )

records = []

for replicate in range(REPLICATES):
    for greek in GREEKS:
        for method in METHODS:
            key, result = run_fixed(key, greek, method, replicate)
            records.append(result)

        key, result = run_adaptive(key, greek, replicate)
        records.append(result)

    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

df = pd.DataFrame(records)
df.to_csv("results/equal_budget_estimates.csv", index=False)

summary = df.groupby(["greek", "method"], as_index=False).agg(
    mean_estimate=("estimate", "mean"),
    bias=("error", "mean"),
    rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    mean_absolute_error=("absolute_error", "mean"),
    mean_standard_error=("standard_error", "mean"),
    mean_production_paths=("production_paths", "mean"),
    mean_elapsed_seconds=("elapsed_seconds", "mean"),
    mean_pilot_seconds=("pilot_seconds", "mean"),
    replicates=("replicate", "nunique"),
)

summary.to_csv("results/equal_budget_summary.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, greek in zip(axes, GREEKS):
    subset = summary[summary["greek"] == greek]
    ax.bar(subset["method"], subset["rmse"])
    ax.set_title(greek)
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("figures/equal_budget_rmse.png", dpi=300)
plt.close(fig)

print(summary.to_string(index=False))
print("\nSaved results/equal_budget_estimates.csv")
print("Saved results/equal_budget_summary.csv")
print("Saved figures/equal_budget_rmse.png")

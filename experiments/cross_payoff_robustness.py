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
from mcadgreeks.estimators.greek_estimators import estimator_samples
from mcadgreeks.estimators.digital_estimators import pathwise_ad_samples, lrm_samples, fd_greek_samples
from mcadgreeks.references.black_scholes import delta_call, vega_call, rho_call
from mcadgreeks.references.digital import delta_cash_digital_call, vega_cash_digital_call, rho_cash_digital_call

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)
GREEKS = ("Delta", "Vega", "Rho")
PAYOFFS = ("EuropeanCall", "CashDigitalCall")
METHODS = ("Pathwise", "LRM", "FD_CRN")
PILOT_SIZES = (2048, 8192)
BATCH_PATHS = 4096
BUDGET_SECONDS = 0.05
REPLICATES = 20
SEED = 20261012

FD_STEPS = {
    "EuropeanCall": {"Delta": 0.1, "Vega": 0.005, "Rho": 0.001},
    "CashDigitalCall": {"Delta": 0.1, "Vega": 0.005, "Rho": 0.01},
}

REFERENCES = {
    "EuropeanCall": {
        "Delta": delta_call(*PARAMETERS),
        "Vega": vega_call(*PARAMETERS),
        "Rho": rho_call(*PARAMETERS),
    },
    "CashDigitalCall": {
        "Delta": delta_cash_digital_call(*PARAMETERS),
        "Vega": vega_cash_digital_call(*PARAMETERS),
        "Rho": rho_cash_digital_call(*PARAMETERS),
    },
}

def call_samples(z, S0, K, r, sigma, T):
    ST = S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    return jnp.exp(-r * T) * jnp.maximum(ST - K, 0.0)

def european_fd_samples(z, greek, h):
    S0, K, r, sigma, T = PARAMETERS

    if greek == "Delta":
        upper = call_samples(z, S0 + h, K, r, sigma, T)
        lower = call_samples(z, S0 - h, K, r, sigma, T)
    elif greek == "Vega":
        upper = call_samples(z, S0, K, r, sigma + h, T)
        lower = call_samples(z, S0, K, r, sigma - h, T)
    else:
        upper = call_samples(z, S0, K, r + h, sigma, T)
        lower = call_samples(z, S0, K, r - h, sigma, T)

    return (upper - lower) / (2.0 * h)

def make_functions(payoff, greek):
    result = {}

    for method in METHODS:
        if payoff == "EuropeanCall":
            if method == "FD_CRN":
                fn = lambda z, g=greek: european_fd_samples(
                    z, g, FD_STEPS[payoff][g]
                )
            else:
                fn = lambda z, g=greek, m=method: estimator_samples(
                    z, g, m, *PARAMETERS
                )
        else:
            if method == "Pathwise":
                fn = lambda z, g=greek: pathwise_ad_samples(
                    z, g, *PARAMETERS
                )
            elif method == "LRM":
                fn = lambda z, g=greek: lrm_samples(
                    z, g, *PARAMETERS
                )
            else:
                fn = lambda z, g=greek: fd_greek_samples(
                    z, g, *PARAMETERS, FD_STEPS[payoff][g]
                )

        result[method] = jax.jit(fn)

    return result

FUNCTIONS = {
    (payoff, greek): make_functions(payoff, greek)
    for payoff in PAYOFFS
    for greek in GREEKS
}

key = jax.random.key(SEED)

for functions in FUNCTIONS.values():
    for fn in functions.values():
        for n in (BATCH_PATHS,) + PILOT_SIZES:
            fn(jnp.zeros((n,), dtype=jnp.float64)).block_until_ready()

for n in (BATCH_PATHS,) + PILOT_SIZES:
    jax.random.normal(jax.random.key(SEED + n), (n,), dtype=jnp.float64).block_until_ready()

def draw_batch(key, fn, n=BATCH_PATHS):
    key, sample_key = jax.random.split(key)
    z = jax.random.normal(sample_key, (n,), dtype=jnp.float64)
    values = fn(z)
    values.block_until_ready()
    return key, np.asarray(values, dtype=np.float64)

def create_record(payoff, greek, method, selected, total, total_sq, n, elapsed, pilot, replicate):
    if n < 2:
        raise RuntimeError("Insufficient production samples")

    estimate = total / n
    variance = max(0.0, (total_sq - total * total / n) / (n - 1))
    standard_error = np.sqrt(variance / n)
    reference = REFERENCES[payoff][greek]

    return {
        "replicate": replicate,
        "payoff": payoff,
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
        "pilot_seconds": pilot,
    }

def run_fixed(key, payoff, greek, method, replicate):
    fn = FUNCTIONS[(payoff, greek)][method]
    start = time.perf_counter()
    total = 0.0
    total_sq = 0.0
    n = 0

    while n == 0 or time.perf_counter() - start < BUDGET_SECONDS:
        key, values = draw_batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += len(values)

    elapsed = time.perf_counter() - start
    row = create_record(
        payoff, greek, "Fixed_" + method, method,
        total, total_sq, n, elapsed, 0.0, replicate
    )
    return key, row

def evaluate_candidates(key, payoff, greek, pilot_size):
    start = time.perf_counter()
    key, pilot_key = jax.random.split(key)
    z = jax.random.normal(pilot_key, (pilot_size,), dtype=jnp.float64)
    z.block_until_ready()
    candidates = []

    for method, fn in FUNCTIONS[(payoff, greek)].items():
        output = fn(z)
        output.block_until_ready()
        samples = np.asarray(output, dtype=np.float64)
        timings = []

        for _ in range(3):
            tick = time.perf_counter()
            output = fn(z)
            output.block_until_ready()
            samples = np.asarray(output, dtype=np.float64)
            timings.append(time.perf_counter() - tick)

        variance = float(np.var(samples, ddof=1))
        seconds_per_path = float(np.median(timings)) / pilot_size

        candidates.append({
            "method": method,
            "pilot_variance": variance,
            "seconds_per_path": seconds_per_path,
            "variance_time_score": variance * seconds_per_path,
        })

    return key, candidates, time.perf_counter() - start

def run_adaptive(key, payoff, greek, pilot_size, constrained, replicate):
    start = time.perf_counter()
    key, candidates, _ = evaluate_candidates(
        key, payoff, greek, pilot_size
    )

    if constrained:
        selected = select_valid_candidate(payoff, greek, candidates)["method"]
        policy = f"Adaptive_Valid_P{pilot_size}"
    else:
        selected = min(candidates, key=lambda x: x["variance_time_score"])["method"]
        policy = f"Adaptive_Any_P{pilot_size}"

    pilot_elapsed = time.perf_counter() - start
    fn = FUNCTIONS[(payoff, greek)][selected]
    total = 0.0
    total_sq = 0.0
    n = 0

    while n == 0 or time.perf_counter() - start < BUDGET_SECONDS:
        key, values = draw_batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += len(values)

    elapsed = time.perf_counter() - start
    row = create_record(
        payoff, greek, policy, selected, total, total_sq,
        n, elapsed, pilot_elapsed, replicate
    )

    return key, row

records = []

for replicate in range(REPLICATES):
    for payoff in PAYOFFS:
        for greek in GREEKS:
            for method in METHODS:
                key, row = run_fixed(key, payoff, greek, method, replicate)
                records.append(row)

            for pilot_size in PILOT_SIZES:
                for constrained in (False, True):
                    key, row = run_adaptive(
                        key, payoff, greek, pilot_size,
                        constrained, replicate
                    )
                    records.append(row)

    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

data = pd.DataFrame(records)
data.to_csv("results/cross_payoff_robustness_estimates.csv", index=False)

summary = data.groupby(
    ["payoff", "greek", "method"], as_index=False
).agg(
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
summary.to_csv("results/cross_payoff_robustness_summary.csv", index=False)

regret_rows = []

for payoff in PAYOFFS:
    for greek in GREEKS:
        subset = summary[
            (summary["payoff"] == payoff) &
            (summary["greek"] == greek)
        ]
        valid_methods = admissible_methods(payoff, greek)
        fixed = subset[
            subset["method"].isin(["Fixed_" + method for method in valid_methods])
        ]
        oracle = fixed.loc[fixed["rmse"].idxmin()]

        for _, row in subset[
            subset["method"].str.startswith("Adaptive_")
        ].iterrows():
            regret_rows.append({
                "payoff": payoff,
                "greek": greek,
                "policy": row["method"],
                "best_valid_fixed": oracle["method"],
                "best_valid_fixed_rmse": oracle["rmse"],
                "policy_rmse": row["rmse"],
                "empirical_rmse_gap": row["rmse"] - oracle["rmse"],
                "mean_pilot_seconds": row["mean_pilot_seconds"],
                "mean_elapsed_seconds": row["mean_elapsed_seconds"],
            })

regret = pd.DataFrame(regret_rows)
regret.to_csv("results/cross_payoff_regret.csv", index=False)

selection = data[data["method"].str.startswith("Adaptive_")].groupby(
    ["payoff", "greek", "method", "selected_estimator"], as_index=False
).size()
selection.to_csv("results/cross_payoff_selection_counts.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

for row_index, payoff in enumerate(PAYOFFS):
    for col_index, greek in enumerate(GREEKS):
        ax = axes[row_index, col_index]
        subset = summary[
            (summary["payoff"] == payoff) &
            (summary["greek"] == greek)
        ]
        ax.bar(subset["method"], subset["rmse"])
        ax.set_title(f"{payoff}: {greek}")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=80)
        ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("figures/cross_payoff_robustness.png", dpi=300)
plt.close(fig)

print("\nROBUSTNESS SUMMARY")
print(summary.to_string(index=False))
print("\nEMPIRICAL REGRET")
print(regret.to_string(index=False))
print("\nSELECTION COUNTS")
print(selection.to_string(index=False))

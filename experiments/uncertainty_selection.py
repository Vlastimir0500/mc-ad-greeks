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

from mcadgreeks.adaptive.uncertainty import choose_candidate
from mcadgreeks.estimators.greek_estimators import estimator_samples
from mcadgreeks.estimators.digital_estimators import pathwise_ad_samples, lrm_samples, fd_greek_samples
from mcadgreeks.references.black_scholes import delta_call, vega_call, rho_call
from mcadgreeks.references.digital import delta_cash_digital_call, vega_cash_digital_call, rho_cash_digital_call

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)
GREEKS = ("Delta", "Vega", "Rho")
PAYOFFS = ("EuropeanCall", "CashDigitalCall")
METHODS = ("Pathwise", "LRM", "FD_CRN")
PILOT_SIZES = (1024, 4096)
BATCH_PATHS = 4096
BUDGET_SECONDS = 0.05
REPLICATES = 10
BOOTSTRAPS = 60
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

def create_fn(payoff, greek, method):
    if payoff == "EuropeanCall":
        if method == "FD_CRN":
            return jax.jit(
                lambda z, g=greek: european_fd_samples(z, g)
            )
        return jax.jit(
            lambda z, g=greek, m=method:
                estimator_samples(z, g, m, *PARAMETERS)
        )

    if method == "Pathwise":
        return jax.jit(
            lambda z, g=greek: pathwise_ad_samples(z, g, *PARAMETERS)
        )
    if method == "LRM":
        return jax.jit(
            lambda z, g=greek: lrm_samples(z, g, *PARAMETERS)
        )
    return jax.jit(
        lambda z, g=greek: fd_greek_samples(
            z, g, *PARAMETERS, FD_STEPS["CashDigitalCall"][g]
        )
    )

def european_fd_samples(z, greek):
    S0, K, r, sigma, T = PARAMETERS
    h = FD_STEPS["EuropeanCall"][greek]

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

FUNCTIONS = {
    (payoff, greek): {
        method: create_fn(payoff, greek, method)
        for method in METHODS
    }
    for payoff in PAYOFFS
    for greek in GREEKS
}

key = jax.random.key(SEED)

for functions in FUNCTIONS.values():
    for fn in functions.values():
        for n in (BATCH_PATHS,) + PILOT_SIZES:
            fn(jnp.zeros((n,), dtype=jnp.float64)).block_until_ready()

def draw_batch(key, fn, n=BATCH_PATHS):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (n,), dtype=jnp.float64)
    values = fn(z)
    values.block_until_ready()
    return key, np.asarray(values, dtype=np.float64)

def score_candidates(key, payoff, greek, pilot_size, rng):
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
            timings.append(time.perf_counter() - tick)

        variance = float(np.var(samples, ddof=1))
        seconds_per_path = float(np.median(timings)) / pilot_size
        indices = rng.integers(
            0, pilot_size, size=(BOOTSTRAPS, pilot_size)
        )
        bootstrap_variances = np.var(samples[indices], axis=1, ddof=1)
        bootstrap_scores = bootstrap_variances * seconds_per_path

        candidates.append({
            "method": method,
            "pilot_variance": variance,
            "seconds_per_path": seconds_per_path,
            "variance_time_score": variance * seconds_per_path,
            "variance_time_lower10": float(np.quantile(bootstrap_scores, 0.10)),
            "variance_time_upper90": float(np.quantile(bootstrap_scores, 0.90)),
        })

    return key, candidates, time.perf_counter() - start

def make_result(payoff, greek, method, selected, total, total_sq, n, elapsed, pilot, replicate, pilot_size):
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
        "pilot_size": pilot_size,
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
    row = make_result(
        payoff, greek, "Fixed_" + method, method,
        total, total_sq, n, elapsed, 0.0, replicate, 0
    )
    return key, row

def run_adaptive(key, payoff, greek, pilot_size, candidates, pilot_seconds, criterion, replicate):
    selected = choose_candidate(
        candidates, payoff, greek, criterion
    )["method"]

    start = time.perf_counter()
    fn = FUNCTIONS[(payoff, greek)][selected]
    total = 0.0
    total_sq = 0.0
    n = 0
    remaining_budget = max(0.0, BUDGET_SECONDS - pilot_seconds)

    while n == 0 or time.perf_counter() - start < remaining_budget:
        key, values = draw_batch(key, fn)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        n += len(values)

    production_seconds = time.perf_counter() - start
    elapsed = pilot_seconds + production_seconds
    labels = {
        "unconstrained_point": f"Adaptive_Any_Point_P{pilot_size}",
        "valid_point": f"Adaptive_Valid_Point_P{pilot_size}",
        "valid_upper90": f"Adaptive_Valid_UCB90_P{pilot_size}",
    }

    row = make_result(
        payoff, greek, labels[criterion], selected,
        total, total_sq, n, elapsed, pilot_seconds, replicate, pilot_size
    )
    return key, row

records = []
selection_records = []
rng = np.random.default_rng(SEED)

for replicate in range(REPLICATES):
    for payoff in PAYOFFS:
        for greek in GREEKS:
            for method in METHODS:
                key, row = run_fixed(key, payoff, greek, method, replicate)
                records.append(row)

            for pilot_size in PILOT_SIZES:
                rng = np.random.default_rng(SEED + replicate * 100000 + pilot_size)
                key, candidates, pilot_seconds = score_candidates(
                    key, payoff, greek, pilot_size, rng
                )

                for candidate in candidates:
                    selection_records.append({
                        "replicate": replicate,
                        "payoff": payoff,
                        "greek": greek,
                        "pilot_size": pilot_size,
                        **candidate,
                    })

                for criterion in (
                    "unconstrained_point",
                    "valid_point",
                    "valid_upper90",
                ):
                    key, row = run_adaptive(
                        key, payoff, greek, pilot_size, candidates,
                        pilot_seconds, criterion, replicate
                    )
                    records.append(row)

    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

data = pd.DataFrame(records)
data.to_csv("results/uncertainty_selection_estimates.csv", index=False)
pd.DataFrame(selection_records).to_csv(
    "results/uncertainty_selection_pilot_scores.csv", index=False
)

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
summary.to_csv("results/uncertainty_selection_summary.csv", index=False)

selection_counts = data[
    data["method"].str.startswith("Adaptive_")
].groupby(
    ["payoff", "greek", "method", "selected_estimator", "pilot_size"],
    as_index=False,
).size()
selection_counts.to_csv(
    "results/uncertainty_selection_counts.csv", index=False
)

regret_rows = []

for payoff in PAYOFFS:
    for greek in GREEKS:
        subset = summary[
            (summary["payoff"] == payoff) &
            (summary["greek"] == greek)
        ]
        from mcadgreeks.adaptive.validity import admissible_methods
        allowed = set(admissible_methods(payoff, greek))
        fixed = subset[
            subset["method"].isin(["Fixed_" + method for method in allowed])
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
regret.to_csv("results/uncertainty_selection_regret.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, payoff in enumerate(PAYOFFS):
    for j, greek in enumerate(GREEKS):
        ax = axes[i, j]
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
fig.savefig("figures/uncertainty_selection_rmse.png", dpi=300)
plt.close(fig)

print("\nUNCERTAINTY-AWARE SELECTION SUMMARY")
print(summary.to_string(index=False))
print("\nEMPIRICAL REGRET")
print(regret.to_string(index=False))
print("\nSELECTION COUNTS")
print(selection_counts.to_string(index=False))

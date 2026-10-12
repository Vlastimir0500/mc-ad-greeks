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

from mcadgreeks.references.digital import (
    price_cash_digital_call,
    delta_cash_digital_call,
    vega_cash_digital_call,
    rho_cash_digital_call,
)
from mcadgreeks.estimators.digital_estimators import (
    mc_cash_digital_price,
    pathwise_delta_samples,
    lrm_samples,
    fd_delta_samples,
)
from mcadgreeks.adaptive.validity import select_valid_candidate

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
PARAMETERS = (S0, K, R, SIGMA, T)
N_PATHS = 200000
REPLICATES = 30
STEPS = [1.0, 0.3, 0.1, 0.03, 0.01]
REFERENCES = {
    "Price": price_cash_digital_call(*PARAMETERS),
    "Delta": delta_cash_digital_call(*PARAMETERS),
    "Vega": vega_cash_digital_call(*PARAMETERS),
    "Rho": rho_cash_digital_call(*PARAMETERS),
}

key = jax.random.key(20261012)
ad_delta = jax.jit(jax.grad(mc_cash_digital_price, argnums=1))
lrm_fn = jax.jit(lambda z: lrm_samples(z, "Delta", *PARAMETERS))
pathwise_fn = jax.jit(lambda z: pathwise_delta_samples(z, *PARAMETERS))
fd_functions = {
    h: jax.jit(lambda z, bump=h: fd_delta_samples(z, *PARAMETERS, bump))
    for h in STEPS
}

rows = []

for replicate in range(REPLICATES):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (N_PATHS,), dtype=jnp.float64)

    ad_value = ad_delta(z, *PARAMETERS)
    ad_value.block_until_ready()
    pathwise_values = np.asarray(pathwise_fn(z))
    lrm_values = np.asarray(lrm_fn(z))

    rows.append({
        "replicate": replicate,
        "method": "PathwiseAD",
        "step_size": 0.0,
        "estimate": float(ad_value),
        "standard_error": float(pathwise_values.std(ddof=1) / np.sqrt(N_PATHS)),
        "reference": REFERENCES["Delta"],
        "paths": N_PATHS,
    })

    rows.append({
        "replicate": replicate,
        "method": "LRM",
        "step_size": 0.0,
        "estimate": float(lrm_values.mean()),
        "standard_error": float(lrm_values.std(ddof=1) / np.sqrt(N_PATHS)),
        "reference": REFERENCES["Delta"],
        "paths": N_PATHS,
    })

    for h, fn in fd_functions.items():
        values = np.asarray(fn(z))
        rows.append({
            "replicate": replicate,
            "method": "FD_CRN",
            "step_size": h,
            "estimate": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(N_PATHS)),
            "reference": REFERENCES["Delta"],
            "paths": N_PATHS,
        })

    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

df = pd.DataFrame(rows)
df["error"] = df["estimate"] - df["reference"]
df["absolute_error"] = df["error"].abs()
df["covered_95"] = df["absolute_error"] <= 1.96 * df["standard_error"]
df.to_csv("results/digital_delta_estimates.csv", index=False)

summary = df.groupby(["method", "step_size"], as_index=False).agg(
    mean_estimate=("estimate", "mean"),
    bias=("error", "mean"),
    rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    mean_absolute_error=("absolute_error", "mean"),
    mean_standard_error=("standard_error", "mean"),
    empirical_coverage_95=("covered_95", "mean"),
    replicates=("replicate", "nunique"),
)

summary.to_csv("results/digital_delta_summary.csv", index=False)

key, pilot_key = jax.random.split(key)
pilot_z = jax.random.normal(pilot_key, (N_PATHS,), dtype=jnp.float64)
candidate_functions = {
    "Pathwise": pathwise_fn,
    "LRM": lrm_fn,
    "FD_CRN": fd_functions[0.1],
}

candidates = []

for method, fn in candidate_functions.items():
    values = fn(pilot_z)
    values.block_until_ready()
    samples = np.asarray(values)
    timings = []

    for _ in range(5):
        start = time.perf_counter()
        result = fn(pilot_z)
        result.block_until_ready()
        timings.append(time.perf_counter() - start)

    variance = float(np.var(samples, ddof=1))
    seconds_per_path = float(np.median(timings)) / N_PATHS

    candidates.append({
        "method": method,
        "pilot_variance": variance,
        "seconds_per_path": seconds_per_path,
        "variance_time_score": variance * seconds_per_path,
    })

naive_choice = min(candidates, key=lambda item: item["variance_time_score"])
valid_choice = select_valid_candidate("CashDigitalCall", "Delta", candidates)

pd.DataFrame(candidates).to_csv("results/digital_selector_diagnostics.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))

for method in ["PathwiseAD", "LRM", "FD_CRN"]:
    subset = summary[summary["method"] == method].sort_values("step_size")
    subset = subset[subset["rmse"] > 0]
    if not subset.empty:
        ax.plot(subset["step_size"], subset["rmse"], marker="o", label=method)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Finite-difference step size")
ax.set_ylabel("RMSE against analytical Delta")
ax.set_title("Cash-or-Nothing Digital Call: Greek Estimation")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figures/digital_delta_validity.png", dpi=300)
plt.close(fig)

print("\nANALYTICAL REFERENCES")
for name, value in REFERENCES.items():
    print(f"{name}: {value:.10f}")

print("\nDELTA BENCHMARK")
print(summary.to_string(index=False))

print("\nNAIVE VARIANCE-TIME CHOICE")
print(naive_choice["method"])

print("\nVALIDITY-CONSTRAINED CHOICE")
print(valid_choice["method"])

print("\nSaved results/digital_delta_estimates.csv")
print("Saved results/digital_delta_summary.csv")
print("Saved results/digital_selector_diagnostics.csv")
print("Saved figures/digital_delta_validity.png")

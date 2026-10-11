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

from mcadgreeks.estimators.autodiff import mc_price_from_z
from mcadgreeks.references.black_scholes import (
    price_call,
    delta_call,
    gamma_call,
    vega_call,
    rho_call,
)

S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0
N_PATHS = 100000
N_REPLICATES = 20
SEED = 20261011
SPOT_STEPS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
VOL_STEPS = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

REFERENCES = {
    "Price": price_call(S0, K, R, SIGMA, T),
    "Delta": delta_call(S0, K, R, SIGMA, T),
    "Gamma": gamma_call(S0, K, R, SIGMA, T),
    "Vega": vega_call(S0, K, R, SIGMA, T),
    "Rho": rho_call(S0, K, R, SIGMA, T),
}

price_fn = jax.jit(mc_price_from_z)
delta_fn = jax.jit(jax.grad(mc_price_from_z, argnums=1))
gamma_fn = jax.jit(jax.grad(jax.grad(mc_price_from_z, argnums=1), argnums=1))
vega_fn = jax.jit(jax.grad(mc_price_from_z, argnums=4))
rho_fn = jax.jit(jax.grad(mc_price_from_z, argnums=3))

records = []
key = jax.random.key(SEED)

def evaluate(fn, *args):
    start = time.perf_counter()
    value = fn(*args)
    value.block_until_ready()
    return float(value), time.perf_counter() - start

def add_record(method, greek, step, replicate, estimate, elapsed):
    reference = REFERENCES[greek]
    records.append({
        "method": method,
        "greek": greek,
        "step_size": step,
        "replicate": replicate,
        "estimate": estimate,
        "reference": reference,
        "error": estimate - reference,
        "absolute_error": abs(estimate - reference),
        "runtime_seconds": elapsed,
    })

for replicate in range(N_REPLICATES):
    key, sample_key = jax.random.split(key)
    z = jax.random.normal(sample_key, (N_PATHS,), dtype=jnp.float64)

    p, pt = evaluate(price_fn, z, S0, K, R, SIGMA, T)
    d, dt = evaluate(delta_fn, z, S0, K, R, SIGMA, T)
    g, gt = evaluate(gamma_fn, z, S0, K, R, SIGMA, T)
    v, vt = evaluate(vega_fn, z, S0, K, R, SIGMA, T)
    rho, rt = evaluate(rho_fn, z, S0, K, R, SIGMA, T)

    add_record("JAX_AD", "Price", 0.0, replicate, p, pt)
    add_record("JAX_AD", "Delta", 0.0, replicate, d, dt)
    add_record("JAX_AD", "Gamma", 0.0, replicate, g, gt)
    add_record("JAX_AD", "Vega", 0.0, replicate, v, vt)
    add_record("JAX_AD", "Rho", 0.0, replicate, rho, rt)

    for h in SPOT_STEPS:
        start = time.perf_counter()
        up, _ = evaluate(price_fn, z, S0 + h, K, R, SIGMA, T)
        down, _ = evaluate(price_fn, z, S0 - h, K, R, SIGMA, T)
        elapsed = time.perf_counter() - start
        add_record("FD_CRN", "Delta", h, replicate, (up - down) / (2.0 * h), elapsed)
        add_record("FD_CRN", "Gamma", h, replicate, (up - 2.0 * p + down) / (h * h), elapsed)

    for h in VOL_STEPS:
        start = time.perf_counter()
        up, _ = evaluate(price_fn, z, S0, K, R, SIGMA + h, T)
        down, _ = evaluate(price_fn, z, S0, K, R, SIGMA - h, T)
        elapsed = time.perf_counter() - start
        add_record("FD_CRN", "Vega", h, replicate, (up - down) / (2.0 * h), elapsed)

    start = time.perf_counter()
    up, _ = evaluate(price_fn, z, S0, K, R + 0.0001, SIGMA, T)
    down, _ = evaluate(price_fn, z, S0, K, R - 0.0001, SIGMA, T)
    add_record("FD_CRN", "Rho", 0.0001, replicate, (up - down) / 0.0002, time.perf_counter() - start)

data = pd.DataFrame(records)
data.to_csv("results/phase2_estimates.csv", index=False)

summary_rows = []
for (method, greek, step), group in data.groupby(["method", "greek", "step_size"], dropna=False):
    estimates = group["estimate"].to_numpy()
    errors = estimates - group["reference"].to_numpy()
    summary_rows.append({
        "method": method,
        "greek": greek,
        "step_size": step,
        "reference": float(group["reference"].iloc[0]),
        "mean_estimate": float(np.mean(estimates)),
        "bias": float(np.mean(errors)),
        "absolute_bias": float(abs(np.mean(errors))),
        "standard_deviation": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
        "replicates": len(group),
        "paths_per_replicate": N_PATHS,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv("results/phase2_summary.csv", index=False)

for greek in ["Delta", "Gamma", "Vega", "Rho"]:
    subset = summary[summary["greek"] == greek]
    fd = subset[(subset["method"] == "FD_CRN")].sort_values("step_size")
    ad = subset[subset["method"] == "JAX_AD"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fd = fd[fd["rmse"] > 0]
    if not fd.empty:
        ax.loglog(fd["step_size"], fd["rmse"], marker="o", label="Finite Difference with CRN")
    if not ad.empty:
        ad_rmse = float(ad["rmse"].iloc[0])
        x_values = fd["step_size"].to_numpy()
        if len(x_values) > 0:
            ax.loglog(x_values, np.full_like(x_values, ad_rmse, dtype=float), linestyle="--", label="JAX AD")
    ax.set_xlabel("Perturbation step size")
    ax.set_ylabel("RMSE against analytical reference")
    ax.set_title(f"{greek}: Finite Difference vs JAX AD")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/phase2_{greek.lower()}_rmse.png", dpi=300)
    plt.close(fig)

status = [
    "PROJECT: Monte Carlo Pricing Sensitivities with Automatic Differentiation (JAX)",
    "PROJECT_COMPLETION: 30%",
    "CURRENT_PHASE: Finite Difference vs Automatic Differentiation",
    "MODEL: Black-Scholes Geometric Brownian Motion",
    "PAYOFF: European Call",
    f"PATHS_PER_REPLICATE: {N_PATHS}",
    f"INDEPENDENT_REPLICATES: {N_REPLICATES}",
    "METHODS: JAX AD, central finite differences, common random numbers",
    "OUTPUTS: results/phase2_estimates.csv, results/phase2_summary.csv",
    "FIGURES: figures/phase2_delta_rmse.png, figures/phase2_gamma_rmse.png, figures/phase2_vega_rmse.png, figures/phase2_rho_rmse.png",
    "NEXT_PHASE: Likelihood-Ratio and Pathwise Sensitivity Estimators",
]
Path("PROJECT_STATUS.txt").write_text("\n".join(status) + "\n", encoding="utf-8")

print(summary.sort_values(["greek", "method", "step_size"]).to_string(index=False))
print("\nPROJECT_COMPLETION: 30%")

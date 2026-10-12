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
    pathwise_ad_samples,
    lrm_samples,
    fd_greek_samples,
)
from mcadgreeks.adaptive.validity import admissible_methods

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
PARAMETERS = (S0, K, R, SIGMA, T)
GREEKS = ("Delta", "Vega", "Rho")
REPLICATES = 30
N_PATHS = 150000
SEED = 20261012
STEPS = {
    "Delta": [1.0, 0.3, 0.1, 0.03, 0.01],
    "Vega": [0.05, 0.02, 0.01, 0.005, 0.001],
    "Rho": [0.1, 0.03, 0.01, 0.003, 0.001],
}
REFERENCES = {
    "Delta": delta_cash_digital_call(*PARAMETERS),
    "Vega": vega_cash_digital_call(*PARAMETERS),
    "Rho": rho_cash_digital_call(*PARAMETERS),
}
ARGUMENTS = {"Delta": 1, "Rho": 3, "Vega": 4}
ad_functions = {
    greek: jax.jit(jax.grad(mc_cash_digital_price, argnums=ARGUMENTS[greek]))
    for greek in GREEKS
}
lrm_functions = {
    greek: jax.jit(lambda z, g=greek: lrm_samples(z, g, *PARAMETERS))
    for greek in GREEKS
}
pathwise_functions = {
    greek: jax.jit(lambda z, g=greek: pathwise_ad_samples(z, g, *PARAMETERS))
    for greek in GREEKS
}
fd_functions = {
    (greek, h): jax.jit(
        lambda z, g=greek, bump=h: fd_greek_samples(
            z, g, *PARAMETERS, bump
        )
    )
    for greek in GREEKS
    for h in STEPS[greek]
}

key = jax.random.key(SEED)
rows = []

for replicate in range(REPLICATES):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (N_PATHS,), dtype=jnp.float64)

    for greek in GREEKS:
        reference = REFERENCES[greek]
        pathwise_samples = np.asarray(pathwise_functions[greek](z))
        ad_estimate = float(ad_functions[greek](z, *PARAMETERS))
        lrm_samples_value = np.asarray(lrm_functions[greek](z))

        estimates = [
            ("PathwiseAD", 0.0, ad_estimate, pathwise_samples),
            ("LRM", 0.0, float(lrm_samples_value.mean()), lrm_samples_value),
        ]

        for h in STEPS[greek]:
            samples = np.asarray(fd_functions[(greek, h)](z))
            estimates.append(
                ("FD_CRN", h, float(samples.mean()), samples)
            )

        for method, h, estimate, samples in estimates:
            se = float(samples.std(ddof=1) / np.sqrt(N_PATHS))
            error = estimate - reference
            rows.append({
                "replicate": replicate,
                "greek": greek,
                "method": method,
                "step_size": h,
                "estimate": estimate,
                "reference": reference,
                "error": error,
                "absolute_error": abs(error),
                "standard_error": se,
                "covered_95": abs(error) <= 1.96 * se,
                "paths": N_PATHS,
            })

    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

df = pd.DataFrame(rows)
df.to_csv("results/digital_all_greeks_estimates.csv", index=False)

summary = df.groupby(
    ["greek", "method", "step_size"], as_index=False
).agg(
    mean_estimate=("estimate", "mean"),
    bias=("error", "mean"),
    rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    mean_absolute_error=("absolute_error", "mean"),
    mean_standard_error=("standard_error", "mean"),
    empirical_coverage_95=("covered_95", "mean"),
    replicates=("replicate", "nunique"),
)
summary.to_csv("results/digital_all_greeks_summary.csv", index=False)

validity_rows = []
for greek in GREEKS:
    allowed = admissible_methods("CashDigitalCall", greek)
    for method in ("Pathwise", "LRM", "FD_CRN"):
        validity_rows.append({
            "payoff": "CashDigitalCall",
            "greek": greek,
            "method": method,
            "admissible": method in allowed,
        })
pd.DataFrame(validity_rows).to_csv(
    "results/digital_validity_matrix.csv", index=False
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, greek in zip(axes, GREEKS):
    subset = summary[summary["greek"] == greek]
    fd = subset[(subset["method"] == "FD_CRN") & (subset["rmse"] > 0)]
    if not fd.empty:
        ax.loglog(fd["step_size"], fd["rmse"], marker="o", label="FD_CRN")
    for method in ("PathwiseAD", "LRM"):
        item = subset[subset["method"] == method]
        if not item.empty:
            ax.axhline(
                float(item["rmse"].iloc[0]),
                linestyle="--",
                label=method,
            )
    ax.set_title(greek)
    ax.set_xlabel("Finite-difference step size")
    ax.set_ylabel("RMSE")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig("figures/digital_all_greeks.png", dpi=300)
plt.close(fig)

print("\nANALYTICAL REFERENCES")
print(f"Price: {price_cash_digital_call(*PARAMETERS):.10f}")
for greek, value in REFERENCES.items():
    print(f"{greek}: {value:.10f}")
print("\nBENCHMARK SUMMARY")
print(summary.to_string(index=False))
print("\nVALIDITY MATRIX")
print(pd.DataFrame(validity_rows).to_string(index=False))

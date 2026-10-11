import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.selector import estimate_greek
from mcadgreeks.estimators.greek_estimators import estimator_samples
from mcadgreeks.references.black_scholes import delta_call, vega_call, rho_call

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
parameters = (S0, K, r, sigma, T)
n_paths = 200000
pilot_paths = 20000
replicates = 10

references = {
    "Delta": delta_call(S0, K, r, sigma, T),
    "Vega": vega_call(S0, K, r, sigma, T),
    "Rho": rho_call(S0, K, r, sigma, T),
}

key = jax.random.key(20261011)
rows = []

for replicate in range(replicates):
    for greek in ("Delta", "Vega", "Rho"):
        key, subkey = jax.random.split(key)
        result = estimate_greek(
            subkey,
            greek,
            "EuropeanCall",
            parameters,
            n_paths=n_paths,
            pilot_paths=pilot_paths,
        )

        rows.append({
            "replicate": replicate,
            "greek": greek,
            "method": "AdaptiveSelected",
            "selected_method": result["selected_method"],
            "estimate": result["estimate"],
            "reference": references[greek],
            "error": result["estimate"] - references[greek],
            "absolute_error": abs(result["estimate"] - references[greek]),
            "standard_error": result["standard_error"],
            "production_seconds": result["production_seconds"],
            "paths": n_paths,
            "pilot_paths": pilot_paths,
        })

        for candidate in result["pilot_candidates"]:
            key, eval_key = jax.random.split(key)
            z = jax.random.normal(eval_key, (n_paths,), dtype=jnp.float64)
            values = estimator_samples(
                z, greek, candidate["method"], *parameters
            )
            values.block_until_ready()
            samples = np.asarray(values)
            estimate = float(samples.mean())

            rows.append({
                "replicate": replicate,
                "greek": greek,
                "method": candidate["method"],
                "selected_method": result["selected_method"],
                "estimate": estimate,
                "reference": references[greek],
                "error": estimate - references[greek],
                "absolute_error": abs(estimate - references[greek]),
                "standard_error": float(samples.std(ddof=1) / np.sqrt(n_paths)),
                "production_seconds": candidate["pilot_seconds"],
                "paths": n_paths,
                "pilot_paths": pilot_paths,
            })

    print(f"Completed replicate {replicate + 1}/{replicates}")

df = pd.DataFrame(rows)
df.to_csv("results/adaptive_estimator_benchmark.csv", index=False)

summary = df.groupby(["greek", "method"], as_index=False).agg(
    mean_estimate=("estimate", "mean"),
    bias=("error", "mean"),
    rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    mean_absolute_error=("absolute_error", "mean"),
    mean_reported_standard_error=("standard_error", "mean"),
    mean_measured_seconds=("production_seconds", "mean"),
    replicate_count=("replicate", "nunique"),
)

summary.to_csv("results/adaptive_estimator_summary.csv", index=False)
print(summary.to_string(index=False))

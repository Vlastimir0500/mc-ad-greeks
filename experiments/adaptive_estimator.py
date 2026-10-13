import platform
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.selector import estimate_greek, _compiled_sample_function
from mcadgreeks.adaptive.validity import admissible_methods
from mcadgreeks.references.black_scholes import delta_call, gamma_call, vega_call, rho_call, delta_put, gamma_put, vega_put, rho_put
from mcadgreeks.references.digital import delta_cash_digital_call, gamma_cash_digital_call, vega_cash_digital_call, rho_cash_digital_call

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)
PAYOFFS = ("EuropeanCall", "EuropeanPut", "CashDigitalCall")
GREEKS = ("Delta", "Gamma", "Vega", "Rho")
METHODS = ("Pathwise", "LRM", "FD_CRN")
N_PATHS = 200000
PILOT_PATHS = 20000
REPLICATES = 10
SEED = 20261013

REFERENCES = {
    "EuropeanCall": {
        "Delta": delta_call(*PARAMETERS),
        "Gamma": gamma_call(*PARAMETERS),
        "Vega": vega_call(*PARAMETERS),
        "Rho": rho_call(*PARAMETERS),
    },
    "EuropeanPut": {
        "Delta": delta_put(*PARAMETERS),
        "Gamma": gamma_put(*PARAMETERS),
        "Vega": vega_put(*PARAMETERS),
        "Rho": rho_put(*PARAMETERS),
    },
    "CashDigitalCall": {
        "Delta": delta_cash_digital_call(*PARAMETERS),
        "Gamma": gamma_cash_digital_call(*PARAMETERS),
        "Vega": vega_cash_digital_call(*PARAMETERS),
        "Rho": rho_cash_digital_call(*PARAMETERS),
    },
}

key = jax.random.key(SEED)
rows = []
compiled = {}

for payoff in PAYOFFS:
    for greek in GREEKS:
        reference = REFERENCES[payoff][greek]
        for method in METHODS:
            fn = _compiled_sample_function(payoff, greek, method, tuple(PARAMETERS))
            pilot_warmup = fn(jnp.zeros((PILOT_PATHS,), dtype=jnp.float64))
            pilot_warmup.block_until_ready()
            production_warmup = fn(jnp.zeros((N_PATHS,), dtype=jnp.float64))
            production_warmup.block_until_ready()
            compiled[(payoff, greek, method)] = fn

for replicate in range(REPLICATES):
    for payoff in PAYOFFS:
        for greek in GREEKS:
            reference = REFERENCES[payoff][greek]
            key, adaptive_key = jax.random.split(key)
            adaptive = estimate_greek(
                adaptive_key,
                greek,
                payoff,
                PARAMETERS,
                n_paths=N_PATHS,
                pilot_paths=PILOT_PATHS,
                criterion="valid_point",
            )
            rows.append({
                "replicate": replicate,
                "payoff": payoff,
                "greek": greek,
                "method": "AdaptiveSelected",
                "selected_method": adaptive["selected_method"],
                "estimate": adaptive["estimate"],
                "reference": reference,
                "error": adaptive["estimate"] - reference,
                "absolute_error": abs(adaptive["estimate"] - reference),
                "standard_error": adaptive["standard_error"],
                "selection_seconds": adaptive["selection_seconds"],
                "production_warmup_seconds": adaptive["production_warmup_seconds"],
                "production_seconds": adaptive["production_seconds"],
                "total_elapsed_seconds": adaptive["total_elapsed_seconds"],
                "paths": N_PATHS,
                "pilot_paths": PILOT_PATHS,
            })

            for method in METHODS:
                key, eval_key = jax.random.split(key)
                fn = compiled[(payoff, greek, method)]
                start = time.perf_counter()
                z = jax.random.normal(eval_key, (N_PATHS,), dtype=jnp.float64)
                values = fn(z)
                values.block_until_ready()
                elapsed = time.perf_counter() - start
                samples = np.asarray(values, dtype=np.float64)
                estimate = float(samples.mean())
                rows.append({
                    "replicate": replicate,
                    "payoff": payoff,
                    "greek": greek,
                    "method": method,
                    "selected_method": adaptive["selected_method"],
                    "estimate": estimate,
                    "reference": reference,
                    "error": estimate - reference,
                    "absolute_error": abs(estimate - reference),
                    "standard_error": float(samples.std(ddof=1) / np.sqrt(N_PATHS)),
                    "selection_seconds": 0.0,
                    "production_warmup_seconds": 0.0,
                    "production_seconds": elapsed,
                    "total_elapsed_seconds": elapsed,
                    "paths": N_PATHS,
                    "pilot_paths": 0,
                })
    print(f"Completed replicate {replicate + 1}/{REPLICATES}")

df = pd.DataFrame(rows)
df.to_csv("results/adaptive_estimator_benchmark.csv", index=False)
summary = df.groupby(["payoff", "greek", "method"], as_index=False).agg(
    mean_estimate=("estimate", "mean"),
    bias=("error", "mean"),
    rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    mean_absolute_error=("absolute_error", "mean"),
    mean_reported_standard_error=("standard_error", "mean"),
    mean_selection_seconds=("selection_seconds", "mean"),
    mean_production_warmup_seconds=("production_warmup_seconds", "mean"),
    mean_production_seconds=("production_seconds", "mean"),
    mean_total_elapsed_seconds=("total_elapsed_seconds", "mean"),
    replicate_count=("replicate", "nunique"),
)
summary.to_csv("results/adaptive_estimator_summary.csv", index=False)
environment = [
    "BENCHMARK_ENVIRONMENT",
    f"PYTHON: {sys.version.split()[0]}",
    f"PLATFORM: {platform.platform()}",
    f"JAX: {jax.__version__}",
    f"NUMPY: {np.__version__}",
    f"PANDAS: {pd.__version__}",
    "JAX_ENABLE_X64: true",
    f"PATHS_PER_ESTIMATE: {N_PATHS}",
    f"PILOT_PATHS: {PILOT_PATHS}",
    f"REPLICATES: {REPLICATES}",
    f"ESTIMATE_ROWS: {len(df)}",
    f"SUMMARY_ROWS: {len(summary)}",
    "VALIDATION: run experiments/validate_benchmark.py after this script",
]
Path("results/adaptive_benchmark_environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
print(summary.to_string(index=False))

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
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
r = 0.05
sigma = 0.20
T = 1.0
seed = 42

path_counts = [1000, 5000, 10000, 50000, 100000, 500000]

key = jax.random.key(seed)

reference_price = price_call(S0, K, r, sigma, T)
reference_delta = delta_call(S0, K, r, sigma, T)
reference_gamma = gamma_call(S0, K, r, sigma, T)
reference_vega = vega_call(S0, K, r, sigma, T)
reference_rho = rho_call(S0, K, r, sigma, T)

rows = []

for n_paths in path_counts:
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, shape=(n_paths,))

    compiled_price = jax.jit(mc_price_from_z)

    start = time.perf_counter()

    price_value = compiled_price(z, S0, K, r, sigma, T)
    delta_value = jax.grad(mc_price_from_z, argnums=1)(
        z, S0, K, r, sigma, T
    )
    vega_value = jax.grad(mc_price_from_z, argnums=4)(
        z, S0, K, r, sigma, T
    )
    rho_value = jax.grad(mc_price_from_z, argnums=3)(
        z, S0, K, r, sigma, T
    )

    price_value.block_until_ready()
    delta_value.block_until_ready()
    vega_value.block_until_ready()
    rho_value.block_until_ready()

    elapsed = time.perf_counter() - start

    rows.append(
        {
            "paths": n_paths,
            "price": float(price_value),
            "delta": float(delta_value),
            "vega": float(vega_value),
            "rho": float(rho_value),
            "price_abs_error": abs(float(price_value) - reference_price),
            "delta_abs_error": abs(float(delta_value) - reference_delta),
            "vega_abs_error": abs(float(vega_value) - reference_vega),
            "rho_abs_error": abs(float(rho_value) - reference_rho),
            "runtime_seconds": elapsed,
        }
    )

results = pd.DataFrame(rows)

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

results.to_csv("results/phase1_results.csv", index=False)

plt.figure()
plt.loglog(results["paths"], results["delta_abs_error"], marker="o")
plt.xlabel("Number of paths")
plt.ylabel("Absolute Delta error")
plt.title("JAX AD Delta convergence")
plt.grid(True)
plt.savefig("figures/delta_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.loglog(results["paths"], results["vega_abs_error"], marker="o")
plt.xlabel("Number of paths")
plt.ylabel("Absolute Vega error")
plt.title("JAX AD Vega convergence")
plt.grid(True)
plt.savefig("figures/vega_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.loglog(results["paths"], results["rho_abs_error"], marker="o")
plt.xlabel("Number of paths")
plt.ylabel("Absolute Rho error")
plt.title("JAX AD Rho convergence")
plt.grid(True)
plt.savefig("figures/rho_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

print(results.to_string(index=False))
print()
print("BLACK-SCHOLES REFERENCE")
print(f"Price: {reference_price:.10f}")
print(f"Delta: {reference_delta:.10f}")
print(f"Gamma: {reference_gamma:.10f}")
print(f"Vega: {reference_vega:.10f}")
print(f"Rho: {reference_rho:.10f}")

import time
import jax
import jax.numpy as jnp
import numpy as np

from mcadgreeks.estimators.greek_estimators import estimator_samples

SUPPORTED = {
    "EuropeanCall": {
        "Delta": ("Pathwise", "LRM"),
        "Vega": ("Pathwise", "LRM"),
        "Rho": ("Pathwise", "LRM"),
    }
}

def select_estimator(key, greek, payoff, parameters, pilot_paths=20000, timing_repeats=3):
    if payoff not in SUPPORTED or greek not in SUPPORTED[payoff]:
        raise ValueError("No validated estimator set for this payoff and Greek")

    S0, K, r, sigma, T = parameters
    candidates = []

    for method in SUPPORTED[payoff][greek]:
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (pilot_paths,), dtype=jnp.float64)
        fn = jax.jit(lambda x: estimator_samples(x, greek, method, S0, K, r, sigma, T))
        values = fn(z)
        values.block_until_ready()

        samples = np.asarray(values)
        times = []

        for _ in range(timing_repeats):
            start = time.perf_counter()
            values = fn(z)
            values.block_until_ready()
            times.append(time.perf_counter() - start)

        variance = float(np.var(samples, ddof=1))
        steady_seconds = float(np.median(times))
        seconds_per_path = steady_seconds / pilot_paths

        candidates.append({
            "method": method,
            "pilot_variance": variance,
            "pilot_seconds": steady_seconds,
            "seconds_per_path": seconds_per_path,
            "variance_time_score": variance * seconds_per_path,
        })

    selected = min(candidates, key=lambda item: item["variance_time_score"])
    return selected, candidates

def estimate_greek(key, greek, payoff, parameters, n_paths=100000, pilot_paths=20000):
    selected, candidates = select_estimator(
        key, greek, payoff, parameters, pilot_paths
    )

    S0, K, r, sigma, T = parameters
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (n_paths,), dtype=jnp.float64)
    method = selected["method"]

    fn = jax.jit(lambda x: estimator_samples(x, greek, method, S0, K, r, sigma, T))
    values = fn(z)
    values.block_until_ready()

    start = time.perf_counter()
    values = fn(z)
    values.block_until_ready()
    elapsed = time.perf_counter() - start

    samples = np.asarray(values)

    return {
        "greek": greek,
        "payoff": payoff,
        "selected_method": method,
        "estimate": float(np.mean(samples)),
        "standard_error": float(np.std(samples, ddof=1) / np.sqrt(n_paths)),
        "pilot_candidates": candidates,
        "production_seconds": elapsed,
        "production_paths": n_paths,
    }

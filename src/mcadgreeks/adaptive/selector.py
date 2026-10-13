import time
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from mcadgreeks.adaptive.uncertainty import choose_candidate
from mcadgreeks.adaptive.validity import admissible_methods
from mcadgreeks.estimators.digital_estimators import fd_greek_samples, lrm_samples, pathwise_ad_samples
from mcadgreeks.estimators.greek_estimators import discounted_call_samples, discounted_put_samples, estimator_samples

SUPPORTED = {
    "EuropeanCall": {
        "Delta": ("Pathwise", "LRM", "FD_CRN"),
        "Gamma": ("Pathwise", "LRM", "FD_CRN"),
        "Vega": ("Pathwise", "LRM", "FD_CRN"),
        "Rho": ("Pathwise", "LRM", "FD_CRN"),
    },
    "EuropeanPut": {
        "Delta": ("Pathwise", "LRM", "FD_CRN"),
        "Gamma": ("Pathwise", "LRM", "FD_CRN"),
        "Vega": ("Pathwise", "LRM", "FD_CRN"),
        "Rho": ("Pathwise", "LRM", "FD_CRN"),
    },
    "CashDigitalCall": {
        "Delta": ("Pathwise", "LRM", "FD_CRN"),
        "Gamma": ("Pathwise", "LRM", "FD_CRN"),
        "Vega": ("Pathwise", "LRM", "FD_CRN"),
        "Rho": ("Pathwise", "LRM", "FD_CRN"),
    },
}

FD_STEPS = {
    "EuropeanCall": {"Delta": 0.1, "Gamma": 0.25, "Vega": 0.005, "Rho": 0.001},
    "EuropeanPut": {"Delta": 0.1, "Gamma": 0.25, "Vega": 0.005, "Rho": 0.001},
    "CashDigitalCall": {"Delta": 0.1, "Gamma": 0.1, "Vega": 0.005, "Rho": 0.01},
}
CRITERIA = {"unconstrained_point", "valid_point", "valid_upper90"}


def _validate_inputs(payoff, greek, parameters, pilot_paths, timing_repeats, bootstrap_replicates, criterion):
    admissible_methods(payoff, greek)
    if criterion not in CRITERIA:
        raise ValueError("Unsupported selection criterion")
    if len(parameters) != 5:
        raise ValueError("Parameters must contain S0, K, r, sigma, and T")
    values = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Parameters must be finite")
    S0, K, r, sigma, T = values
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, K, sigma, and T must be positive")
    if not isinstance(pilot_paths, (int, np.integer)) or pilot_paths < 2:
        raise ValueError("pilot_paths must be an integer of at least 2")
    if not isinstance(timing_repeats, (int, np.integer)) or timing_repeats < 1:
        raise ValueError("timing_repeats must be an integer of at least 1")
    if not isinstance(bootstrap_replicates, (int, np.integer)) or bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be an integer of at least 2")


def _european_fd_samples(payoff, z, greek, S0, K, r, sigma, T):
    h = FD_STEPS[payoff][greek]
    sample_fn = discounted_call_samples if payoff == "EuropeanCall" else discounted_put_samples
    if greek == "Delta":
        h = min(h, S0 * 0.25)
        upper = sample_fn(z, S0 + h, K, r, sigma, T)
        lower = sample_fn(z, S0 - h, K, r, sigma, T)
    elif greek == "Vega":
        h = min(h, sigma * 0.25)
        upper = sample_fn(z, S0, K, r, sigma + h, T)
        lower = sample_fn(z, S0, K, r, sigma - h, T)
    elif greek == "Gamma":
        h = min(h, S0 * 0.25)
        upper = sample_fn(z, S0 + h, K, r, sigma, T)
        center = sample_fn(z, S0, K, r, sigma, T)
        lower = sample_fn(z, S0 - h, K, r, sigma, T)
        return (upper - 2.0 * center + lower) / (h * h)
    elif greek == "Rho":
        upper = sample_fn(z, S0, K, r + h, sigma, T)
        lower = sample_fn(z, S0, K, r - h, sigma, T)
    else:
        raise ValueError("Unsupported Greek")
    return (upper - lower) / (2.0 * h)


def estimator_function(payoff, greek, method, parameters):
    S0, K, r, sigma, T = parameters
    if payoff in ("EuropeanCall", "EuropeanPut"):
        if method in ("Pathwise", "LRM"):
            return lambda z: estimator_samples(z, greek, method, S0, K, r, sigma, T, payoff=payoff)
        if method == "FD_CRN":
            return lambda z: _european_fd_samples(payoff, z, greek, S0, K, r, sigma, T)
    elif payoff == "CashDigitalCall":
        if method == "Pathwise":
            return lambda z: pathwise_ad_samples(z, greek, S0, K, r, sigma, T)
        if method == "LRM":
            return lambda z: lrm_samples(z, greek, S0, K, r, sigma, T)
        if method == "FD_CRN":
            h = FD_STEPS["CashDigitalCall"][greek]
            if greek == "Vega":
                h = min(h, sigma * 0.25)
            elif greek in ("Delta", "Gamma"):
                h = min(h, S0 * 0.25)
            return lambda z: fd_greek_samples(z, greek, S0, K, r, sigma, T, h)
    raise ValueError("Unsupported payoff, Greek, or estimator method")

@lru_cache(maxsize=128)
def _compiled_sample_function(payoff, greek, method, parameters):
    return jax.jit(estimator_function(payoff, greek, method, parameters))


def select_estimator(
    key,
    greek,
    payoff,
    parameters,
    pilot_paths=20000,
    timing_repeats=3,
    criterion="valid_point",
    bootstrap_replicates=60,
):
    _validate_inputs(
        payoff, greek, parameters, pilot_paths, timing_repeats, bootstrap_replicates, criterion
    )
    selection_start = time.perf_counter()
    pilot_key = jax.random.fold_in(key, 0)
    z = jax.random.normal(pilot_key, (pilot_paths,), dtype=jnp.float64)
    methods = SUPPORTED[payoff][greek]
    candidates = []
    sample_cache = {}

    for method in methods:
        fn = _compiled_sample_function(payoff, greek, method, tuple(parameters))
        values = fn(z)
        values.block_until_ready()
        samples = np.asarray(values, dtype=np.float64)
        sample_cache[method] = samples
        timings = []

        for _ in range(timing_repeats):
            tick = time.perf_counter()
            values = fn(z)
            values.block_until_ready()
            timings.append(time.perf_counter() - tick)

        variance = float(np.var(samples, ddof=1))
        steady_seconds = float(np.median(timings))
        seconds_per_path = steady_seconds / pilot_paths
        candidates.append({
            "method": method,
            "pilot_variance": variance,
            "pilot_seconds": steady_seconds,
            "seconds_per_path": seconds_per_path,
            "variance_time_score": variance * seconds_per_path,
        })

    if criterion == "valid_upper90":
        rng = np.random.default_rng(20261013)
        for candidate in candidates:
            samples = sample_cache[candidate["method"]]
            indices = rng.integers(0, pilot_paths, size=(bootstrap_replicates, pilot_paths))
            bootstrap_variances = np.var(samples[indices], axis=1, ddof=1)
            bootstrap_scores = bootstrap_variances * candidate["seconds_per_path"]
            candidate["variance_time_lower10"] = float(np.quantile(bootstrap_scores, 0.10))
            candidate["variance_time_upper90"] = float(np.quantile(bootstrap_scores, 0.90))

    selected = choose_candidate(candidates, payoff, greek, criterion)
    selection_seconds = time.perf_counter() - selection_start
    for candidate in candidates:
        candidate["selection_seconds"] = selection_seconds
        candidate["selection_criterion"] = criterion
    return selected, candidates


def estimate_greek(
    key,
    greek,
    payoff,
    parameters,
    n_paths=100000,
    pilot_paths=20000,
    criterion="valid_point",
    timing_repeats=3,
    bootstrap_replicates=60,
):
    call_start = time.perf_counter()
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least 2")
    selected, candidates = select_estimator(
        key,
        greek,
        payoff,
        parameters,
        pilot_paths=pilot_paths,
        timing_repeats=timing_repeats,
        criterion=criterion,
        bootstrap_replicates=bootstrap_replicates,
    )
    production_key = jax.random.fold_in(key, 1)
    fn = _compiled_sample_function(payoff, greek, selected["method"], tuple(parameters))
    warmup_start = time.perf_counter()
    warmup_values = fn(jnp.zeros((n_paths,), dtype=jnp.float64))
    warmup_values.block_until_ready()
    production_warmup_seconds = time.perf_counter() - warmup_start
    start = time.perf_counter()
    z = jax.random.normal(production_key, (n_paths,), dtype=jnp.float64)
    values = fn(z)
    values.block_until_ready()
    production_seconds = time.perf_counter() - start
    samples = np.asarray(values, dtype=np.float64)
    selection_seconds = selected["selection_seconds"]
    total_elapsed_seconds = time.perf_counter() - call_start

    return {
        "greek": greek,
        "payoff": payoff,
        "selected_method": selected["method"],
        "selection_criterion": criterion,
        "estimate": float(np.mean(samples)),
        "standard_error": float(np.std(samples, ddof=1) / np.sqrt(n_paths)),
        "pilot_candidates": candidates,
        "selection_seconds": selection_seconds,
        "production_warmup_seconds": production_warmup_seconds,
        "production_seconds": production_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "production_paths": n_paths,
    }

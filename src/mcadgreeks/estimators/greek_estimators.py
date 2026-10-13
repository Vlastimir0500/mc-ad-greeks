import jax.numpy as jnp


def simulate_terminal(z, S0, r, sigma, T):
    return S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)


def discounted_call_samples(z, S0, K, r, sigma, T):
    ST = simulate_terminal(z, S0, r, sigma, T)
    return jnp.exp(-r * T) * jnp.maximum(ST - K, 0.0)


def discounted_put_samples(z, S0, K, r, sigma, T):
    ST = simulate_terminal(z, S0, r, sigma, T)
    return jnp.exp(-r * T) * jnp.maximum(K - ST, 0.0)


def discounted_payoff_samples(z, S0, K, r, sigma, T, payoff):
    if payoff == "EuropeanCall":
        return discounted_call_samples(z, S0, K, r, sigma, T)
    if payoff == "EuropeanPut":
        return discounted_put_samples(z, S0, K, r, sigma, T)
    raise ValueError("Unsupported payoff")


def pathwise_delta_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    ST = simulate_terminal(z, S0, r, sigma, T)
    if payoff == "EuropeanCall":
        slope = (ST > K) * ST / S0
    elif payoff == "EuropeanPut":
        slope = -(ST < K).astype(ST.dtype) * ST / S0
    else:
        raise ValueError("Unsupported payoff")
    return jnp.exp(-r * T) * slope


def pathwise_gamma_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    return jnp.zeros_like(z)


def lrm_gamma_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    value = discounted_payoff_samples(z, S0, K, r, sigma, T, payoff)
    score = z / (S0 * sigma * jnp.sqrt(T))
    score_derivative = -1.0 / (S0**2 * sigma**2 * T) - z / (S0**2 * sigma * jnp.sqrt(T))
    return value * (score**2 + score_derivative)


def pathwise_vega_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    ST = simulate_terminal(z, S0, r, sigma, T)
    dST = ST * (-sigma * T + jnp.sqrt(T) * z)
    if payoff == "EuropeanCall":
        slope = (ST > K) * dST
    elif payoff == "EuropeanPut":
        slope = -(ST < K).astype(ST.dtype) * dST
    else:
        raise ValueError("Unsupported payoff")
    return jnp.exp(-r * T) * slope


def pathwise_rho_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    ST = simulate_terminal(z, S0, r, sigma, T)
    value = jnp.maximum(ST - K, 0.0) if payoff == "EuropeanCall" else jnp.maximum(K - ST, 0.0)
    if payoff == "EuropeanCall":
        slope = (ST > K) * ST
    elif payoff == "EuropeanPut":
        slope = -(ST < K).astype(ST.dtype) * ST
    else:
        raise ValueError("Unsupported payoff")
    return jnp.exp(-r * T) * T * (slope - value)


def lrm_delta_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    value = discounted_payoff_samples(z, S0, K, r, sigma, T, payoff)
    return value * z / (S0 * sigma * jnp.sqrt(T))


def lrm_vega_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    value = discounted_payoff_samples(z, S0, K, r, sigma, T, payoff)
    score = (z**2 - 1.0) / sigma - z * jnp.sqrt(T)
    return value * score


def lrm_rho_samples(z, S0, K, r, sigma, T, payoff="EuropeanCall"):
    value = discounted_payoff_samples(z, S0, K, r, sigma, T, payoff)
    return value * (z * jnp.sqrt(T) / sigma - T)


def estimator_samples(z, greek, method, S0, K, r, sigma, T, payoff="EuropeanCall"):
    estimators = {
        ("Delta", "Pathwise"): pathwise_delta_samples,
        ("Gamma", "Pathwise"): pathwise_gamma_samples,
        ("Gamma", "LRM"): lrm_gamma_samples,
        ("Delta", "LRM"): lrm_delta_samples,
        ("Vega", "Pathwise"): pathwise_vega_samples,
        ("Vega", "LRM"): lrm_vega_samples,
        ("Rho", "Pathwise"): pathwise_rho_samples,
        ("Rho", "LRM"): lrm_rho_samples,
    }
    return estimators[(greek, method)](z, S0, K, r, sigma, T, payoff)

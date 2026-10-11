import jax.numpy as jnp

def simulate_terminal(z, S0, r, sigma, T):
    return S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)

def discounted_call_samples(z, S0, K, r, sigma, T):
    ST = simulate_terminal(z, S0, r, sigma, T)
    return jnp.exp(-r * T) * jnp.maximum(ST - K, 0.0)

def pathwise_delta_samples(z, S0, K, r, sigma, T):
    ST = simulate_terminal(z, S0, r, sigma, T)
    return jnp.exp(-r * T) * (ST > K) * ST / S0

def pathwise_vega_samples(z, S0, K, r, sigma, T):
    ST = simulate_terminal(z, S0, r, sigma, T)
    dST = ST * (-sigma * T + jnp.sqrt(T) * z)
    return jnp.exp(-r * T) * (ST > K) * dST

def pathwise_rho_samples(z, S0, K, r, sigma, T):
    ST = simulate_terminal(z, S0, r, sigma, T)
    payoff = jnp.maximum(ST - K, 0.0)
    return jnp.exp(-r * T) * T * ((ST > K) * ST - payoff)

def lrm_delta_samples(z, S0, K, r, sigma, T):
    payoff = discounted_call_samples(z, S0, K, r, sigma, T)
    return payoff * z / (S0 * sigma * jnp.sqrt(T))

def lrm_vega_samples(z, S0, K, r, sigma, T):
    payoff = discounted_call_samples(z, S0, K, r, sigma, T)
    score = (z**2 - 1.0) / sigma - z * jnp.sqrt(T)
    return payoff * score

def lrm_rho_samples(z, S0, K, r, sigma, T):
    payoff = discounted_call_samples(z, S0, K, r, sigma, T)
    return payoff * (z * jnp.sqrt(T) / sigma - T)

def estimator_samples(z, greek, method, S0, K, r, sigma, T):
    estimators = {
        ("Delta", "Pathwise"): pathwise_delta_samples,
        ("Delta", "LRM"): lrm_delta_samples,
        ("Vega", "Pathwise"): pathwise_vega_samples,
        ("Vega", "LRM"): lrm_vega_samples,
        ("Rho", "Pathwise"): pathwise_rho_samples,
        ("Rho", "LRM"): lrm_rho_samples,
    }
    return estimators[(greek, method)](z, S0, K, r, sigma, T)

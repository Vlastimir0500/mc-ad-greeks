import jax
import jax.numpy as jnp

def mc_cash_digital_price(z, S0, K, r, sigma, T):
    ST = S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    return jnp.exp(-r * T) * jnp.mean((ST > K).astype(jnp.float64))

def pathwise_delta_samples(z, S0, K, r, sigma, T):
    ST = S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    return jnp.zeros_like(ST)

def lrm_samples(z, greek, S0, K, r, sigma, T):
    ST = S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    payoff = jnp.exp(-r * T) * (ST > K).astype(jnp.float64)

    if greek == "Delta":
        score = z / (S0 * sigma * jnp.sqrt(T))
    elif greek == "Vega":
        score = (z**2 - 1.0) / sigma - z * jnp.sqrt(T)
    elif greek == "Rho":
        score = z * jnp.sqrt(T) / sigma - T
    else:
        raise ValueError("Unsupported Greek")

    return payoff * score

def fd_delta_samples(z, S0, K, r, sigma, T, h):
    factor = jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    upper = (S0 + h) * factor
    lower = (S0 - h) * factor
    upper_payoff = (upper > K).astype(jnp.float64)
    lower_payoff = (lower > K).astype(jnp.float64)
    return jnp.exp(-r * T) * (upper_payoff - lower_payoff) / (2.0 * h)

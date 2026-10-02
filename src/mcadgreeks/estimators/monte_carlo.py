import jax
import jax.numpy as jnp

def mc_call_price(key, S0, K, r, sigma, T, n_paths):
    z = jax.random.normal(key, shape=(n_paths,))
    ST = S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    payoff = jnp.maximum(ST - K, 0.0)
    return jnp.exp(-r * T) * jnp.mean(payoff)

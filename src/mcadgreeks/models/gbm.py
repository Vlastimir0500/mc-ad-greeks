import jax
import jax.numpy as jnp

def terminal_price(s0, r, sigma, T, z):
    return s0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)

def generate_terminal_prices(key, s0, r, sigma, T, n_paths):
    z = jax.random.normal(key, shape=(n_paths,))
    return terminal_price(s0, r, sigma, T, z)

import jax
import jax.numpy as jnp

def mc_price_from_z(z, S0, K, r, sigma, T):
    ST = S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)
    payoff = jnp.maximum(ST - K, 0.0)
    return jnp.exp(-r * T) * jnp.mean(payoff)

def gradients(z, S0, K, r, sigma, T):
    return jax.grad(mc_price_from_z, argnums=(1, 4, 3))(z, S0, K, r, sigma, T)

def hessian_S0(z, S0, K, r, sigma, T):
    return jax.grad(jax.grad(mc_price_from_z, argnums=1), argnums=1)(
        z, S0, K, r, sigma, T
    )

def jacobian(z, S0, K, r, sigma, T):
    f = lambda x: mc_price_from_z(z, x[0], K, x[1], x[2], T)
    x = jnp.array([S0, r, sigma])
    return jax.jacrev(f)(x)

import jax.numpy as jnp


def terminal_price(z, S0, r, sigma, T):
    return S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)


def discounted_digital_samples(z, S0, K, r, sigma, T):
    ST = terminal_price(z, S0, r, sigma, T)
    return jnp.exp(-r * T) * (ST > K).astype(jnp.float64)


def mc_cash_digital_price(z, S0, K, r, sigma, T):
    return jnp.mean(discounted_digital_samples(z, S0, K, r, sigma, T))


def pathwise_ad_samples(z, greek, S0, K, r, sigma, T):
    payoff = discounted_digital_samples(z, S0, K, r, sigma, T)
    if greek in ("Delta", "Gamma", "Vega"):
        return jnp.zeros_like(payoff)
    if greek == "Rho":
        return -T * payoff
    raise ValueError("Unsupported Greek")


def pathwise_delta_samples(z, S0, K, r, sigma, T):
    return pathwise_ad_samples(z, "Delta", S0, K, r, sigma, T)


def lrm_samples(z, greek, S0, K, r, sigma, T):
    payoff = discounted_digital_samples(z, S0, K, r, sigma, T)
    if greek == "Delta":
        score = z / (S0 * sigma * jnp.sqrt(T))
    elif greek == "Vega":
        score = (z**2 - 1.0) / sigma - z * jnp.sqrt(T)
    elif greek == "Rho":
        score = z * jnp.sqrt(T) / sigma - T
    elif greek == "Gamma":
        first_score = z / (S0 * sigma * jnp.sqrt(T))
        score_derivative = -1.0 / (S0**2 * sigma**2 * T) - z / (S0**2 * sigma * jnp.sqrt(T))
        score = first_score**2 + score_derivative
    else:
        raise ValueError("Unsupported Greek")
    return payoff * score


def fd_greek_samples(z, greek, S0, K, r, sigma, T, h):
    if h <= 0:
        raise ValueError("Step size must be positive")
    if greek == "Delta":
        upper = discounted_digital_samples(z, S0 + h, K, r, sigma, T)
        lower = discounted_digital_samples(z, S0 - h, K, r, sigma, T)
    elif greek == "Gamma":
        if S0 - h <= 0:
            raise ValueError("Spot step produces a nonpositive value")
        upper = discounted_digital_samples(z, S0 + h, K, r, sigma, T)
        center = discounted_digital_samples(z, S0, K, r, sigma, T)
        lower = discounted_digital_samples(z, S0 - h, K, r, sigma, T)
        return (upper - 2.0 * center + lower) / (h * h)
    elif greek == "Vega":
        if sigma - h <= 0:
            raise ValueError("Volatility step produces a nonpositive value")
        upper = discounted_digital_samples(z, S0, K, r, sigma + h, T)
        lower = discounted_digital_samples(z, S0, K, r, sigma - h, T)
    elif greek == "Rho":
        upper = discounted_digital_samples(z, S0, K, r + h, sigma, T)
        lower = discounted_digital_samples(z, S0, K, r - h, sigma, T)
    else:
        raise ValueError("Unsupported Greek")
    return (upper - lower) / (2.0 * h)


def fd_delta_samples(z, S0, K, r, sigma, T, h):
    return fd_greek_samples(z, "Delta", S0, K, r, sigma, T, h)

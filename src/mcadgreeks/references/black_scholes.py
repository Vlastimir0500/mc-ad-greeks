import numpy as np
from scipy.stats import norm


def price_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def delta_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def gamma_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def rho_call(S, K, r, sigma, T):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def price_put(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta_put(S, K, r, sigma, T):
    return delta_call(S, K, r, sigma, T) - 1.0


def gamma_put(S, K, r, sigma, T):
    return gamma_call(S, K, r, sigma, T)


def vega_put(S, K, r, sigma, T):
    return vega_call(S, K, r, sigma, T)


def rho_put(S, K, r, sigma, T):
    return rho_call(S, K, r, sigma, T) - K * T * np.exp(-r * T)

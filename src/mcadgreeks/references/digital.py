import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma, and T must be positive")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def price_cash_digital_call(S, K, r, sigma, T):
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return float(np.exp(-r * T) * norm.cdf(d2))


def delta_cash_digital_call(S, K, r, sigma, T):
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return float(np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T)))


def vega_cash_digital_call(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return float(-np.exp(-r * T) * norm.pdf(d2) * d1 / sigma)


def rho_cash_digital_call(S, K, r, sigma, T):
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return float(np.exp(-r * T) * (norm.pdf(d2) * np.sqrt(T) / sigma - T * norm.cdf(d2)))


def gamma_cash_digital_call(S, K, r, sigma, T):
    _, d2 = _d1_d2(S, K, r, sigma, T)
    scale = sigma * np.sqrt(T)
    return float(-np.exp(-r * T) * norm.pdf(d2) * (1.0 + d2 / scale) / (S**2 * scale))

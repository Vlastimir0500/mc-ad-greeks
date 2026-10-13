import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.references.black_scholes import price_call, delta_call, gamma_call, vega_call, rho_call, price_put, delta_put, gamma_put, vega_put, rho_put


def test_black_scholes_call():
    assert np.isclose(price_call(100, 100, 0.05, 0.20, 1.0), 10.4505835722, atol=1e-8)


def test_black_scholes_delta():
    assert np.isclose(delta_call(100, 100, 0.05, 0.20, 1.0), 0.6368306512, atol=1e-8)


def test_black_scholes_gamma():
    assert np.isclose(gamma_call(100, 100, 0.05, 0.20, 1.0), 0.0187620173, atol=1e-8)


def test_black_scholes_vega():
    assert np.isclose(vega_call(100, 100, 0.05, 0.20, 1.0), 37.52403469, atol=1e-6)


def test_black_scholes_rho():
    assert np.isclose(rho_call(100, 100, 0.05, 0.20, 1.0), 53.23248155, atol=1e-6)


def test_black_scholes_put_parity():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    assert np.isclose(price_put(S, K, r, sigma, T), 5.5735260223, atol=1e-8)
    assert np.isclose(delta_put(S, K, r, sigma, T), -0.3631693488, atol=1e-8)
    assert np.isclose(gamma_put(S, K, r, sigma, T), gamma_call(S, K, r, sigma, T), atol=1e-12)
    assert np.isclose(vega_put(S, K, r, sigma, T), vega_call(S, K, r, sigma, T), atol=1e-12)
    assert np.isclose(rho_put(S, K, r, sigma, T), -41.8904609047, atol=1e-6)

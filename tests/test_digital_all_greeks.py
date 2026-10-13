import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.references.digital import (
    delta_cash_digital_call,
    vega_cash_digital_call,
    rho_cash_digital_call,
)
from mcadgreeks.estimators.digital_estimators import (
    mc_cash_digital_price,
    pathwise_ad_samples,
    lrm_samples,
    fd_greek_samples,
)
from mcadgreeks.adaptive.validity import admissible_methods

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)
REFERENCES = {
    "Delta": delta_cash_digital_call(*PARAMETERS),
    "Vega": vega_cash_digital_call(*PARAMETERS),
    "Rho": rho_cash_digital_call(*PARAMETERS),
}
ARGUMENTS = {"Delta": 1, "Rho": 3, "Vega": 4}

@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_pathwise_samples_match_jax_derivative(greek):
    z = jax.random.normal(jax.random.key(51), (20000,), dtype=jnp.float64)
    fn = jax.grad(mc_cash_digital_price, argnums=ARGUMENTS[greek])
    differentiated = float(fn(z, *PARAMETERS))
    samples = pathwise_ad_samples(z, greek, *PARAMETERS)
    assert np.isclose(differentiated, float(jnp.mean(samples)), atol=1e-10)

@pytest.mark.parametrize(
    "greek,tolerance",
    [("Delta", 0.0003), ("Vega", 0.05), ("Rho", 0.04)],
)
def test_lrm_matches_analytical_digital_greek(greek, tolerance):
    z = jax.random.normal(jax.random.key(52), (250000,), dtype=jnp.float64)
    values = lrm_samples(z, greek, *PARAMETERS)
    estimate = float(jnp.mean(values))
    assert abs(estimate - REFERENCES[greek]) < tolerance

@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_digital_pathwise_is_not_admissible(greek):
    valid = admissible_methods("CashDigitalCall", greek)
    assert valid == ("LRM",)

@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_finite_difference_samples_are_finite(greek):
    z = jax.random.normal(jax.random.key(53), (10000,), dtype=jnp.float64)
    h = 0.01
    values = fd_greek_samples(z, greek, *PARAMETERS, h)
    assert values.shape == z.shape
    assert np.all(np.isfinite(np.asarray(values)))

def test_zero_pathwise_delta_is_not_true_delta():
    z = jax.random.normal(jax.random.key(54), (50000,), dtype=jnp.float64)
    values = pathwise_ad_samples(z, "Delta", *PARAMETERS)
    assert float(jnp.mean(values)) == 0.0
    assert REFERENCES["Delta"] > 0.0

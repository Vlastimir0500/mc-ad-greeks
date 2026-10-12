import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.references.digital import (
    price_cash_digital_call,
    delta_cash_digital_call,
)
from mcadgreeks.estimators.digital_estimators import (
    mc_cash_digital_price,
    lrm_samples,
    pathwise_delta_samples,
)
from mcadgreeks.adaptive.validity import (
    admissible_methods,
    select_valid_candidate,
)

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)

def test_digital_analytical_reference_is_positive():
    assert price_cash_digital_call(*PARAMETERS) > 0
    assert delta_cash_digital_call(*PARAMETERS) > 0.015

def test_pathwise_digital_delta_is_degenerate():
    z = jax.random.normal(jax.random.key(41), (50000,), dtype=jnp.float64)
    value = jax.grad(mc_cash_digital_price, argnums=1)(z, *PARAMETERS)
    assert float(value) == 0.0
    assert delta_cash_digital_call(*PARAMETERS) > 0.0

def test_lrm_digital_delta_matches_reference():
    z = jax.random.normal(jax.random.key(42), (250000,), dtype=jnp.float64)
    values = lrm_samples(z, "Delta", *PARAMETERS)
    estimate = float(jnp.mean(values))
    assert abs(estimate - delta_cash_digital_call(*PARAMETERS)) < 0.0003

def test_digital_pathwise_is_not_admissible():
    valid = admissible_methods("CashDigitalCall", "Delta")
    assert "Pathwise" not in valid
    assert "LRM" in valid

def test_validity_gate_rejects_zero_variance_pathwise():
    candidates = [
        {"method": "Pathwise", "variance_time_score": 0.0},
        {"method": "LRM", "variance_time_score": 0.1},
    ]
    selected = select_valid_candidate("CashDigitalCall", "Delta", candidates)
    assert selected["method"] == "LRM"

def test_unknown_payoff_is_rejected():
    with pytest.raises(ValueError):
        admissible_methods("Unknown", "Delta")

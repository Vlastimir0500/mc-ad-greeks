import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.selector import select_estimator, estimate_greek
from mcadgreeks.estimators.greek_estimators import estimator_samples
from mcadgreeks.references.black_scholes import delta_call, vega_call, rho_call

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)


@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
@pytest.mark.parametrize("method", ["Pathwise", "LRM"])
def test_estimator_samples_are_finite(greek, method):
    z = jax.random.normal(jax.random.key(42), (10000,), dtype=jnp.float64)
    values = estimator_samples(z, greek, method, *PARAMETERS)
    assert values.shape == (10000,)
    assert np.all(np.isfinite(np.asarray(values)))


@pytest.mark.parametrize(
    "greek,reference,tolerance",
    [
        ("Delta", delta_call(*PARAMETERS), 0.02),
        ("Vega", vega_call(*PARAMETERS), 1.0),
        ("Rho", rho_call(*PARAMETERS), 1.5),
    ],
)
@pytest.mark.parametrize("method", ["Pathwise", "LRM"])
def test_estimators_match_analytic_reference(greek, reference, tolerance, method):
    z = jax.random.normal(jax.random.key(2026), (250000,), dtype=jnp.float64)
    values = estimator_samples(z, greek, method, *PARAMETERS)
    estimate = float(jnp.mean(values))
    assert abs(estimate - reference) < tolerance


@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_adaptive_selector_returns_supported_method(greek):
    selected, candidates = select_estimator(
        jax.random.key(10), greek, "EuropeanCall", PARAMETERS, pilot_paths=2000,
        timing_repeats=1
    )
    assert selected["method"] in ("Pathwise", "LRM", "FD_CRN")
    assert len(candidates) == 3
    assert np.isfinite(selected["variance_time_score"])


def test_unsupported_greek_is_rejected():
    with pytest.raises(ValueError):
        select_estimator(
            jax.random.key(10), "Theta", "EuropeanCall", PARAMETERS, pilot_paths=1000
        )


def test_adaptive_estimator_supports_digital_payoff():
    result = estimate_greek(
        jax.random.key(11), "Delta", "CashDigitalCall", PARAMETERS,
        n_paths=5000, pilot_paths=1000, timing_repeats=1
    )
    assert np.isfinite(result["estimate"])
    assert result["selected_method"] in ("LRM", "FD_CRN")

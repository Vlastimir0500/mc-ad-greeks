import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.selector import estimate_greek, select_estimator
from mcadgreeks.adaptive.validity import admissible_methods
from mcadgreeks.estimators.greek_estimators import estimator_samples
from mcadgreeks.estimators.digital_estimators import lrm_samples
from mcadgreeks.references.black_scholes import delta_put, gamma_call, gamma_put, rho_put, vega_put
from mcadgreeks.references.digital import gamma_cash_digital_call

PARAMETERS = (100.0, 100.0, 0.05, 0.20, 1.0)
GREEKS = ("Delta", "Gamma", "Vega", "Rho")
EUROPEAN_PAYOFFS = ("EuropeanCall", "EuropeanPut")
ALL_PAYOFFS = ("EuropeanCall", "EuropeanPut", "CashDigitalCall")


@pytest.mark.parametrize("payoff", EUROPEAN_PAYOFFS)
@pytest.mark.parametrize("greek", GREEKS)
def test_european_selector_includes_all_implemented_estimators(payoff, greek):
    selected, candidates = select_estimator(
        jax.random.key(20261013), greek, payoff, PARAMETERS,
        pilot_paths=2048, timing_repeats=1
    )
    assert selected["method"] in ("Pathwise", "LRM", "FD_CRN")
    assert {candidate["method"] for candidate in candidates} == {"Pathwise", "LRM", "FD_CRN"}
    assert all(np.isfinite(candidate["variance_time_score"]) for candidate in candidates)


@pytest.mark.parametrize("greek", GREEKS)
def test_digital_valid_selection_never_chooses_pathwise(greek):
    selected, candidates = select_estimator(
        jax.random.key(20261014), greek, "CashDigitalCall", PARAMETERS,
        pilot_paths=2048, timing_repeats=1, criterion="valid_point"
    )
    assert selected["method"] in admissible_methods("CashDigitalCall", greek)
    assert selected["method"] == "LRM"
    assert {candidate["method"] for candidate in candidates} == {"Pathwise", "LRM", "FD_CRN"}


@pytest.mark.parametrize("payoff", ALL_PAYOFFS)
def test_gamma_lrm_estimator_matches_analytic_reference(payoff):
    z = jax.random.normal(jax.random.key(20261018), (250000,), dtype=jax.numpy.float64)
    if payoff in EUROPEAN_PAYOFFS:
        values = estimator_samples(z, "Gamma", "LRM", *PARAMETERS, payoff=payoff)
        reference = gamma_call(*PARAMETERS) if payoff == "EuropeanCall" else gamma_put(*PARAMETERS)
    else:
        values = lrm_samples(z, "Gamma", *PARAMETERS)
        reference = gamma_cash_digital_call(*PARAMETERS)
    mean = float(jax.numpy.mean(values))
    standard_error = float(jax.numpy.std(values, ddof=1) / np.sqrt(len(z)))
    assert abs(mean - reference) <= 5.0 * standard_error + 1e-8


@pytest.mark.parametrize("payoff", ALL_PAYOFFS)
def test_gamma_pathwise_is_not_admissible(payoff):
    assert "Pathwise" not in admissible_methods(payoff, "Gamma")
    assert "LRM" in admissible_methods(payoff, "Gamma")
    assert "FD_CRN" in admissible_methods(payoff, "Gamma")


def test_unconstrained_digital_delta_can_select_invalid_zero_variance_pathwise():
    selected, candidates = select_estimator(
        jax.random.key(20261015), "Delta", "CashDigitalCall", PARAMETERS,
        pilot_paths=2048, timing_repeats=1, criterion="unconstrained_point"
    )
    pathwise = next(item for item in candidates if item["method"] == "Pathwise")
    assert pathwise["pilot_variance"] == 0.0
    assert selected["method"] == "Pathwise"


def test_upper90_uses_bootstrap_score_and_applies_validity_filter():
    selected, candidates = select_estimator(
        jax.random.key(20261016), "Delta", "CashDigitalCall", PARAMETERS,
        pilot_paths=1024, timing_repeats=1, criterion="valid_upper90", bootstrap_replicates=8
    )
    assert selected["method"] in admissible_methods("CashDigitalCall", "Delta")
    assert all("variance_time_upper90" in candidate for candidate in candidates)
    assert all(candidate["variance_time_upper90"] >= 0 for candidate in candidates)


@pytest.mark.parametrize("payoff", ALL_PAYOFFS)
@pytest.mark.parametrize("greek", GREEKS)
def test_estimate_greek_returns_finite_estimate_and_separates_selection_cost(payoff, greek):
    result = estimate_greek(
        jax.random.key(20261017), greek, payoff, PARAMETERS,
        n_paths=4096, pilot_paths=1024, timing_repeats=1
    )
    assert np.isfinite(result["estimate"])
    assert np.isfinite(result["standard_error"])
    assert result["standard_error"] >= 0
    assert result["selection_seconds"] >= 0
    assert result["production_warmup_seconds"] >= 0
    assert result["production_seconds"] >= 0
    assert result["total_elapsed_seconds"] >= (
        result["selection_seconds"] + result["production_warmup_seconds"] + result["production_seconds"]
    )
    assert result["production_paths"] == 4096
    assert result["selected_method"] in admissible_methods(payoff, greek)


@pytest.mark.parametrize("parameters", [
    (0.0, 100.0, 0.05, 0.2, 1.0),
    (100.0, 100.0, 0.05, 0.0, 1.0),
    (100.0, 100.0, 0.05, 0.2, 0.0),
    (100.0, 100.0, float("nan"), 0.2, 1.0),
])
def test_selector_rejects_invalid_model_parameters(parameters):
    with pytest.raises(ValueError):
        select_estimator(
            jax.random.key(1), "Delta", "EuropeanCall", parameters,
            pilot_paths=100, timing_repeats=1
        )


def test_selector_rejects_invalid_path_counts():
    with pytest.raises(ValueError):
        select_estimator(
            jax.random.key(1), "Delta", "EuropeanCall", PARAMETERS,
            pilot_paths=1
        )
    with pytest.raises(ValueError):
        estimate_greek(
            jax.random.key(1), "Delta", "EuropeanCall", PARAMETERS,
            n_paths=1, pilot_paths=100
        )


def test_selector_rejects_unknown_criterion_before_running_pilot():
    with pytest.raises(ValueError):
        select_estimator(
            jax.random.key(1), "Delta", "EuropeanCall", PARAMETERS,
            pilot_paths=100, criterion="invalid"
        )


@pytest.mark.parametrize("greek", GREEKS)
def test_european_put_pathwise_estimators_match_analytic_reference(greek):
    z = jax.random.normal(jax.random.key(20261020), (250000,), dtype=jax.numpy.float64)
    if greek == "Gamma":
        values = estimator_samples(z, greek, "LRM", *PARAMETERS, payoff="EuropeanPut")
        reference = gamma_put(*PARAMETERS)
    else:
        values = estimator_samples(z, greek, "Pathwise", *PARAMETERS, payoff="EuropeanPut")
        reference = {
            "Delta": delta_put(*PARAMETERS),
            "Vega": vega_put(*PARAMETERS),
            "Rho": rho_put(*PARAMETERS),
        }[greek]
    mean = float(jax.numpy.mean(values))
    standard_error = float(jax.numpy.std(values, ddof=1) / np.sqrt(len(z)))
    assert abs(mean - reference) <= 5.0 * standard_error + 1e-8


def test_gamma_finite_difference_step_stays_positive_for_small_spot():
    parameters = (0.1, 0.2, 0.01, 0.2, 1.0)
    selected, candidates = select_estimator(
        jax.random.key(20261019), "Gamma", "CashDigitalCall", parameters,
        pilot_paths=512, timing_repeats=1, criterion="valid_point"
    )
    assert selected["method"] == "LRM"
    assert all(np.isfinite(candidate["variance_time_score"]) for candidate in candidates)


@pytest.mark.parametrize("greek", GREEKS)
def test_digital_validity_rules_only_allow_lrm(greek):
    assert admissible_methods("CashDigitalCall", greek) == ("LRM",)

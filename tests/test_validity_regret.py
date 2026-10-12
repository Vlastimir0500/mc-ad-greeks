import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.validity import admissible_methods, select_valid_candidate

@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_digital_screen_excludes_pathwise(greek):
    assert "Pathwise" not in admissible_methods("CashDigitalCall", greek)

@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_digital_screen_keeps_lrm(greek):
    assert "LRM" in admissible_methods("CashDigitalCall", greek)

@pytest.mark.parametrize("greek", ["Delta", "Vega", "Rho"])
def test_validity_selector_avoids_zero_score_pathwise(greek):
    candidates = [
        {"method": "Pathwise", "variance_time_score": 0.0},
        {"method": "LRM", "variance_time_score": 0.2},
        {"method": "FD_CRN", "variance_time_score": 0.3},
    ]
    chosen = select_valid_candidate("CashDigitalCall", greek, candidates)
    assert chosen["method"] in ("LRM", "FD_CRN")

def test_no_valid_candidate_raises():
    with pytest.raises(ValueError):
        select_valid_candidate(
            "CashDigitalCall",
            "Delta",
            [{"method": "Pathwise", "variance_time_score": 0.0}],
        )

def test_european_call_pathwise_remains_available():
    assert "Pathwise" in admissible_methods("EuropeanCall", "Delta")

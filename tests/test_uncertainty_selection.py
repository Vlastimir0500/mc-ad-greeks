import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcadgreeks.adaptive.uncertainty import choose_candidate

def test_unconstrained_selection_can_choose_pathwise():
    candidates = [
        {"method": "Pathwise", "variance_time_score": 0.0, "variance_time_upper90": 0.0},
        {"method": "LRM", "variance_time_score": 0.5, "variance_time_upper90": 0.6},
    ]
    chosen = choose_candidate(
        candidates, "CashDigitalCall", "Delta", "unconstrained_point"
    )
    assert chosen["method"] == "Pathwise"

def test_valid_point_selection_excludes_pathwise():
    candidates = [
        {"method": "Pathwise", "variance_time_score": 0.0, "variance_time_upper90": 0.0},
        {"method": "LRM", "variance_time_score": 0.5, "variance_time_upper90": 0.6},
        {"method": "FD_CRN", "variance_time_score": 0.7, "variance_time_upper90": 0.8},
    ]
    chosen = choose_candidate(
        candidates, "CashDigitalCall", "Delta", "valid_point"
    )
    assert chosen["method"] in ("LRM", "FD_CRN")

def test_upper_bound_selection_uses_upper_score():
    candidates = [
        {"method": "Pathwise", "variance_time_score": 0.0, "variance_time_upper90": 0.0},
        {"method": "LRM", "variance_time_score": 0.1, "variance_time_upper90": 0.7},
        {"method": "FD_CRN", "variance_time_score": 0.2, "variance_time_upper90": 0.3},
    ]
    chosen = choose_candidate(
        candidates, "EuropeanCall", "Delta", "valid_upper90"
    )
    assert chosen["method"] == "FD_CRN"

def test_unknown_criterion_raises():
    with pytest.raises(ValueError):
        choose_candidate([], "EuropeanCall", "Delta", "unknown")

def test_empty_valid_candidate_set_raises():
    with pytest.raises(ValueError):
        choose_candidate(
            [{"method": "Pathwise", "variance_time_score": 0.0}],
            "CashDigitalCall",
            "Delta",
            "valid_point",
        )

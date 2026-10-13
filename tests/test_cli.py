import json

from mcadgreeks import cli


def test_cli_outputs_json_and_forwards_arguments(monkeypatch, capsys):
    captured = {}

    def fake_estimate(key, greek, payoff, parameters, **kwargs):
        captured["greek"] = greek
        captured["payoff"] = payoff
        captured["parameters"] = parameters
        captured["kwargs"] = kwargs
        return {
            "selected_method": "LRM",
            "selection_criterion": kwargs["criterion"],
            "estimate": 0.25,
            "standard_error": 0.01,
            "production_paths": kwargs["n_paths"],
            "selection_seconds": 0.02,
            "production_warmup_seconds": 0.01,
            "production_seconds": 0.03,
            "total_elapsed_seconds": 0.06,
            "pilot_candidates": [
                {
                    "method": "LRM",
                    "pilot_variance": 0.5,
                    "seconds_per_path": 0.00001,
                    "variance_time_score": 0.000005,
                }
            ],
        }

    monkeypatch.setattr(cli, "estimate_greek", fake_estimate)
    result = cli.main([
        "--payoff", "CashDigitalCall",
        "--greek", "Gamma",
        "--spot", "105",
        "--strike", "100",
        "--paths", "5000",
        "--pilot-paths", "512",
        "--criterion", "valid_upper90",
        "--seed", "42",
    ])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["payoff"] == "CashDigitalCall"
    assert output["greek"] == "Gamma"
    assert output["selected_method"] == "LRM"
    assert output["production_paths"] == 5000
    assert captured["parameters"] == (105.0, 100.0, 0.05, 0.2, 1.0)
    assert captured["kwargs"]["pilot_paths"] == 512
    assert captured["kwargs"]["criterion"] == "valid_upper90"
    assert captured["kwargs"]["n_paths"] == 5000


def test_cli_rejects_unsupported_greek():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["--greek", "Theta"])

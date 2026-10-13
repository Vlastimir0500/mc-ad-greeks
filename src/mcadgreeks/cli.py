import argparse
import json

import jax
jax.config.update("jax_enable_x64", True)

from mcadgreeks.adaptive.selector import estimate_greek


def build_parser():
    parser = argparse.ArgumentParser(prog="mcadgreeks")
    parser.add_argument("--payoff", choices=("EuropeanCall", "EuropeanPut", "CashDigitalCall"), default="EuropeanCall")
    parser.add_argument("--greek", choices=("Delta", "Gamma", "Vega", "Rho"), default="Delta")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--volatility", type=float, default=0.2)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--paths", type=int, default=100000)
    parser.add_argument("--pilot-paths", type=int, default=20000)
    parser.add_argument("--criterion", choices=("unconstrained_point", "valid_point", "valid_upper90"), default="valid_point")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=60)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    parameters = (args.spot, args.strike, args.rate, args.volatility, args.maturity)

    try:
        result = estimate_greek(
            jax.random.key(args.seed),
            args.greek,
            args.payoff,
            parameters,
            n_paths=args.paths,
            pilot_paths=args.pilot_paths,
            criterion=args.criterion,
            timing_repeats=args.timing_repeats,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except ValueError as error:
        parser.error(str(error))

    candidates = []
    for item in result["pilot_candidates"]:
        candidate = {
            "method": item["method"],
            "pilot_variance": float(item["pilot_variance"]),
            "seconds_per_path": float(item["seconds_per_path"]),
            "variance_time_score": float(item["variance_time_score"]),
        }
        if "variance_time_upper90" in item:
            candidate["variance_time_upper90"] = float(item["variance_time_upper90"])
        candidates.append(candidate)

    output = {
        "payoff": args.payoff,
        "greek": args.greek,
        "parameters": {
            "spot": args.spot,
            "strike": args.strike,
            "rate": args.rate,
            "volatility": args.volatility,
            "maturity": args.maturity,
        },
        "selected_method": result["selected_method"],
        "criterion": result["selection_criterion"],
        "estimate": float(result["estimate"]),
        "standard_error": float(result["standard_error"]),
        "production_paths": result["production_paths"],
        "selection_seconds": float(result["selection_seconds"]),
        "production_warmup_seconds": float(result["production_warmup_seconds"]),
        "production_seconds": float(result["production_seconds"]),
        "total_elapsed_seconds": float(result["total_elapsed_seconds"]),
        "candidates": candidates,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0

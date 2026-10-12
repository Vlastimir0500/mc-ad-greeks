from .validity import admissible_methods

def choose_candidate(candidates, payoff, greek, criterion):
    keys = {
        "unconstrained_point": "variance_time_score",
        "valid_point": "variance_time_score",
        "valid_upper90": "variance_time_upper90",
    }

    if criterion not in keys:
        raise ValueError("Unsupported selection criterion")

    eligible = candidates
    if criterion != "unconstrained_point":
        allowed = set(admissible_methods(payoff, greek))
        eligible = [item for item in candidates if item["method"] in allowed]

    if not eligible:
        raise ValueError("No eligible estimator candidates")

    return min(eligible, key=lambda item: item[keys[criterion]])

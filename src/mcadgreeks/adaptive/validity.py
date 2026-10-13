ADMISSIBILITY = {
    "EuropeanCall": {
        "Delta": ("Pathwise", "LRM", "FD_CRN"),
        "Gamma": ("LRM", "FD_CRN"),
        "Vega": ("Pathwise", "LRM", "FD_CRN"),
        "Rho": ("Pathwise", "LRM", "FD_CRN"),
    },
    "EuropeanPut": {
        "Delta": ("Pathwise", "LRM", "FD_CRN"),
        "Gamma": ("LRM", "FD_CRN"),
        "Vega": ("Pathwise", "LRM", "FD_CRN"),
        "Rho": ("Pathwise", "LRM", "FD_CRN"),
    },
    "CashDigitalCall": {
        "Delta": ("LRM", "FD_CRN"),
        "Gamma": ("LRM", "FD_CRN"),
        "Vega": ("LRM", "FD_CRN"),
        "Rho": ("LRM", "FD_CRN"),
    },
}


def admissible_methods(payoff, greek):
    if payoff not in ADMISSIBILITY or greek not in ADMISSIBILITY[payoff]:
        raise ValueError("No admissibility rules exist for this payoff and Greek")
    return ADMISSIBILITY[payoff][greek]


def select_valid_candidate(payoff, greek, candidates):
    valid = set(admissible_methods(payoff, greek))
    eligible = [item for item in candidates if item["method"] in valid]
    if not eligible:
        raise ValueError("No admissible estimator candidates were provided")
    return min(eligible, key=lambda item: item["variance_time_score"])

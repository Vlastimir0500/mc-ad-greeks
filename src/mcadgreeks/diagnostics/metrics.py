import numpy as np

def rmse(values, reference):
    values = np.asarray(values)
    return float(np.sqrt(np.mean((values - reference) ** 2)))

def absolute_error(value, reference):
    return float(abs(value - reference))

def relative_error(value, reference):
    return float(abs(value - reference) / max(abs(reference), 1e-15))

def standard_error(values):
    values = np.asarray(values)
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))

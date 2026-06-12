import numpy as np


def compute_basic_stats(returns):
    mean = np.mean(returns)
    std = np.std(returns)

    skew = np.mean((returns - mean) ** 3) / (std ** 3 + 1e-12)
    kurt = np.mean((returns - mean) ** 4) / (std ** 4 + 1e-12)
    kurt -= 3

    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurt
    }
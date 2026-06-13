import numpy as np


def compute_hurst(returns):
    """
    Full-sample Hurst via Rescaled Range (R/S) analysis.
    Returns None if insufficient data or estimation fails.
    """
    if len(returns) < 100:
        return None

    # only use window sizes smaller than the series length
    window_sizes = [w for w in [16, 32, 64, 128, 256] if w < len(returns)]

    rs_values    = []
    valid_windows = []

    for w in window_sizes:
        chunks = len(returns) // w
        vals   = []

        for i in range(chunks):
            segment = returns[i*w:(i+1)*w]

            if len(segment) < 2:
                continue

            mean   = np.mean(segment)
            dev    = segment - mean
            cumsum = np.cumsum(dev)

            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(segment, ddof=1)   # ddof=1: unbiased std

            if S > 0:
                vals.append(R / S)

        if vals:
            rs_values.append(np.mean(vals))
            valid_windows.append(w)

    if len(rs_values) < 2:
        return None

    try:
        hurst, _ = np.polyfit(np.log(valid_windows), np.log(rs_values), 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # sanity clamp: Hurst must be in (0, 1) — discard garbage fits
    if not (0.0 < hurst < 1.0):
        return None

    return hurst


def compute_rolling_hurst(returns, window=256, step=10):
    """
    Rolling Hurst exponent over a sliding window.

    Args:
        returns : array-like of log returns
        window  : lookback size per Hurst calculation
        step    : how many candles to advance between calculations

    Returns:
        np.array of Hurst values, or None if insufficient data
    """
    returns = np.asarray(returns)   # ensure numpy array, not pandas Series

    if len(returns) < window:
        return None

    hursts = []

    for i in range(0, len(returns) - window, step):
        segment = returns[i:i + window]
        h = compute_hurst(segment)
        if h is not None:
            hursts.append(h)

    if len(hursts) < 2:
        return None

    return np.array(hursts)
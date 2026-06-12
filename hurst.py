import numpy as np


def compute_hurst(returns):
    """Full-sample Hurst (keep your existing one)"""

    if len(returns) < 100:
        return None

    window_sizes = [16, 32, 64, 128, 256]

    rs_values = []
    valid_windows = []

    for w in window_sizes:
        if w >= len(returns):
            continue

        chunks = len(returns) // w
        vals = []

        for i in range(chunks):
            segment = returns[i*w:(i+1)*w]

            if len(segment) < 2:
                continue

            mean = np.mean(segment)
            dev = segment - mean
            cumsum = np.cumsum(dev)

            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(segment)

            if S > 0:
                vals.append(R / S)

        if vals:
            rs_values.append(np.mean(vals))
            valid_windows.append(w)

    if len(rs_values) < 2:
        return None

    hurst, _ = np.polyfit(np.log(valid_windows), np.log(rs_values), 1)

    return hurst


# ---------------------------------------------------
# 🔥 NEW: Rolling Hurst
# ---------------------------------------------------

def compute_rolling_hurst(returns, window=256, step=100):
    """
    Returns time series of Hurst values.
    window = lookback size
    step = how often we recalc
    """

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
    print("Size of Hursts Array: ", len(hursts))
    return np.array(hursts)
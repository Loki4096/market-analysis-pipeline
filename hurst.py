import numpy as np
from warehouse import load_from_db


def compute_hurst(symbol: str):
    df = load_from_db(symbol)

    if df is None or len(df) < 100:
        return None

    returns = df["close"].pct_change().dropna().values

    if len(returns) < 100:
        return None

    window_sizes = [16, 32, 64, 128, 256]

    rs_values = []
    valid_windows = []

    for window in window_sizes:

        if window >= len(returns):
            continue

        n_segments = len(returns) // window

        rs_segment_values = []

        for i in range(n_segments):
            segment = returns[i * window:(i + 1) * window]

            mean = np.mean(segment)

            deviations = segment - mean
            cumulative = np.cumsum(deviations)

            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(segment)

            if S > 0:
                rs_segment_values.append(R / S)

        if rs_segment_values:
            rs_values.append(np.mean(rs_segment_values))
            valid_windows.append(window)

    if len(rs_values) < 2:
        return None

    hurst, _ = np.polyfit(
        np.log(valid_windows),
        np.log(rs_values),
        1
    )

    if hurst < 0.40:
        interpretation = "Strong mean reversion"

    elif hurst < 0.45:
        interpretation = "Moderate mean reversion"

    elif hurst < 0.55:
        interpretation = "Random walk / no clear edge"

    elif hurst < 0.60:
        interpretation = "Mild trend persistence"

    else:
        interpretation = "Strong trend persistence"

    return {
        "hurst": hurst,
        "interpretation": interpretation
    }
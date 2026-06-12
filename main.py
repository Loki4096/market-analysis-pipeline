from fetch import fetch_ohlcv
from warehouse import save_to_db, init_db, ticker_exists
from data import AssetData
from analysis import compute_basic_stats
from hurst import compute_hurst, compute_rolling_hurst
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# HURST INTERPRETATION
# -----------------------------
def interpret_hurst(h):
    if h is None:
        return "No Hurst signal available."

    if h < 0.40:
        return "Strong mean reversion — choppy, fading structure dominates."
    elif h < 0.45:
        return "Moderate mean reversion — weak reversal bias."
    elif h < 0.55:
        return "Random walk — no exploitable structure."
    elif h < 0.60:
        return "Mild trend persistence — weak momentum regime."
    else:
        return "Strong trend persistence — momentum regime dominant."


# -----------------------------
# ROLLING HURST ENGINE
# -----------------------------
def rolling_hurst_metrics(rolling):
    latest = rolling[-1]
    mean = np.mean(rolling)
    std = np.std(rolling)

    trend_confirmed = (mean > 0.55 and latest > 0.55)
    delta_h = latest - mean
    pct_trending = np.mean(rolling > 0.55)

    x = np.arange(len(rolling))
    slope = np.polyfit(x, rolling, 1)[0]

    return {
        "latest": latest,
        "mean": mean,
        "std": std,
        "trend_confirmed": trend_confirmed,
        "delta_h": delta_h,
        "pct_trending": pct_trending,
        "slope": slope
    }


# -----------------------------
# INTERPRETATION
# -----------------------------
def interpret_rolling(metrics):
    msg = []

    if metrics["trend_confirmed"]:
        msg.append("Trend regime confirmed (statistically aligned).")
    else:
        msg.append("No stable trend regime.")

    if metrics["delta_h"] > 0:
        msg.append("Regime strengthening (trend structure building).")
    else:
        msg.append("Regime weakening (trend structure fading).")

    if metrics["pct_trending"] > 0.6:
        msg.append("Majority of history in trend regime.")
    elif metrics["pct_trending"] < 0.4:
        msg.append("Mostly non-trending / noisy regime.")

    if metrics["slope"] > 0:
        msg.append("Hurst drifting upward → improving trend structure.")
    else:
        msg.append("Hurst drifting downward → degrading trend structure.")

    return "\n".join(msg)


# -----------------------------
# MAIN
# -----------------------------
def main():
    init_db()

    symbol = input("Ticker: ").upper()
    print(f"\nChecking {symbol}...")

    if not ticker_exists(symbol):
        print("Fetching from Binance...")
        df = fetch_ohlcv(symbol)
        save_to_db(df, symbol)
    else:
        print("Loaded from DB")

    asset = AssetData(symbol)

    if not asset.load():
        print("Not enough data")
        return

    stats = compute_basic_stats(asset.returns)

    hurst = compute_hurst(asset.returns)
    rolling = compute_rolling_hurst(asset.returns, window=256, step=10)

    metrics = None

    # -----------------------------
    # PLOT
    # -----------------------------
    if rolling is not None:

        metrics = rolling_hurst_metrics(rolling)

        plt.style.use("dark_background")
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # HURST
        ax1.plot(rolling, linewidth=2.5, color="white")
        ax1.axhline(0.55, linestyle="--", linewidth=2, color="lime")
        ax1.axhline(0.45, linestyle="--", linewidth=2, color="red")

        ax1.set_ylabel("Hurst", color="white")
        ax1.tick_params(axis='y', labelcolor="white")

        # PRICE (SAFE ALIGNMENT)
        ax2 = ax1.twinx()

        df = asset.df if hasattr(asset, "df") else None

        if df is not None:
            prices = df["close"].values

            # match lengths safely
            min_len = min(len(prices), len(rolling) * 10)
            prices = prices[-min_len:]

            ax2.plot(
                np.linspace(0, len(rolling), len(prices)),
                prices,
                linewidth=2,
                color="cyan",
                alpha=0.6
            )

        ax2.set_ylabel("Price", color="cyan")
        ax2.tick_params(axis='y', labelcolor="cyan")

        ax1.set_title(f"{symbol} — Rolling Hurst vs Price")
        ax1.set_xlabel("Rolling Index")
        ax1.grid(alpha=0.2)

        plt.show(block=False)
        plt.pause(0.1)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print(f"\n{symbol}")

    print(f"\nMean: {stats['mean']:.6f}")
    print(f"Std: {stats['std']:.6f}")
    print(f"Skew: {stats['skew']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")

    if hurst:
        print(f"\nFull Hurst: {hurst:.4f}")
        print("Interpretation:", interpret_hurst(hurst))

    if metrics is not None:
        print("\nRolling Hurst Stats:")
        print(f"Latest H: {metrics['latest']:.3f}")
        print(f"Mean H:   {metrics['mean']:.3f}")
        print(f"Std H:    {metrics['std']:.3f}")

        print("\nRegime Signals:")
        print(f"Trend Confirmed: {metrics['trend_confirmed']}")
        print(f"ΔH:              {metrics['delta_h']:.4f}")
        print(f"% Trending:      {metrics['pct_trending']:.2%}")
        print(f"Slope:           {metrics['slope']:.6f}")

        print("\nInterpretation:")
        print(interpret_rolling(metrics))

    plt.show()


if __name__ == "__main__":
    main()
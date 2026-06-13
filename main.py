import matplotlib
matplotlib.use("TkAgg")

from fetch import fetch_ohlcv
from warehouse import save_to_db, init_db, ticker_exists
from data import AssetData
from analysis import compute_basic_stats
from hurst import compute_hurst, compute_rolling_hurst
from hmm import run_hmm, regime_summary
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
ROLLING_WINDOW = 256
ROLLING_STEP   = 10


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
    mean   = np.mean(rolling)
    std    = np.std(rolling)

    trend_confirmed = (mean > 0.55 and latest > 0.55)
    delta_h         = latest - mean
    pct_trending    = np.mean(rolling > 0.55)

    x     = np.arange(len(rolling))
    slope = np.polyfit(x, rolling, 1)[0]

    return {
        "latest":          latest,
        "mean":            mean,
        "std":             std,
        "trend_confirmed": trend_confirmed,
        "delta_h":         delta_h,
        "pct_trending":    pct_trending,
        "slope":           slope
    }


# -----------------------------
# ROLLING HURST INTERPRETATION
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
# MOMENTUM ENGINE
# -----------------------------
def compute_momentum(prices, lookback):
    if len(prices) <= lookback:
        return None
    return (prices[-1] / prices[-lookback - 1]) - 1


def classify_trend(m20, m50, m100):
    if None in [m20, m50, m100]:
        return "Insufficient data"
    if m20 > 0 and m50 > 0 and m100 > 0:
        if m20 > m50 > m100:
            return "Strong Uptrend (accelerating)"
        return "Uptrend"
    if m20 < 0 and m50 < 0 and m100 < 0:
        if m20 < m50 < m100:
            return "Strong Downtrend (accelerating)"
        return "Downtrend"
    if m20 > 0 and m50 > 0 and m100 < 0:
        return "Potential Bullish Reversal"
    if m20 < 0 and m50 < 0 and m100 > 0:
        return "Potential Bearish Reversal"
    return "Mixed / Transitional Structure"


def trend_strength_score(m20, m50, m100):
    score = 0
    if m20  is not None and m20  > 0: score += 1
    if m50  is not None and m50  > 0: score += 1
    if m100 is not None and m100 > 0: score += 1
    return score


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

    prices_full = asset.df["close"].values
    mom20  = compute_momentum(prices_full, 20)
    mom50  = compute_momentum(prices_full, 50)
    mom100 = compute_momentum(prices_full, 100)

    trend_direction = classify_trend(mom20, mom50, mom100)
    trend_score     = trend_strength_score(mom20, mom50, mom100)

    hurst   = compute_hurst(asset.returns)
    rolling = compute_rolling_hurst(asset.returns, window=ROLLING_WINDOW, step=ROLLING_STEP)

    # HMM
    print("Fitting HMM regime model...")
    hmm_ok       = False
    state_means  = None
    trans_matrix = None

    try:
        label_map, trans_matrix, state_means, _ = run_hmm(asset)
        hmm_ok = True
    except Exception as e:
        print(f"HMM failed: {e}")

    metrics = None

    # -----------------------------
    # BUILD PLOT (show at end)
    # -----------------------------
    if rolling is not None:
        metrics = rolling_hurst_metrics(rolling)

        plt.style.use("dark_background")
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.plot(rolling, linewidth=2.5, color="white")
        ax1.axhline(0.55, linestyle="--", linewidth=2, color="lime")
        ax1.axhline(0.45, linestyle="--", linewidth=2, color="red")

        ax1.set_ylabel("Hurst", color="white")
        ax1.tick_params(axis='y', labelcolor="white")

        ax2 = ax1.twinx()

        df = asset.df if hasattr(asset, "df") else None

        if df is not None:
            prices        = df["close"].values
            price_indices = np.arange(len(prices)) / ROLLING_STEP
            ax2.plot(price_indices, prices, linewidth=2, color="cyan", alpha=0.6)

        if hmm_ok and asset.regimes is not None:
            regime_colours = {
                "low_vol":     "#00ff0022",
                "high_vol":    "#ff000022",
                "extreme_vol": "#ff880033"
            }
            regimes        = asset.regimes
            regime_indices = np.arange(len(regimes)) / ROLLING_STEP

            for i in range(len(regimes) - 1):
                colour = regime_colours.get(regimes[i], "#ffffff11")
                ax1.axvspan(regime_indices[i], regime_indices[i + 1], color=colour, linewidth=0)

        ax2.set_ylabel("Price", color="cyan")
        ax2.tick_params(axis='y', labelcolor="cyan")

        ax1.set_xlim(0, len(rolling))
        ax1.set_title(
            f"{symbol} — Rolling Hurst vs Price  "
            f"[window={ROLLING_WINDOW}, step={ROLLING_STEP}]"
        )
        ax1.set_xlabel("Rolling Index")
        ax1.grid(alpha=0.2)
        plt.tight_layout()

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print(f"\n{'=' * 40}")
    print(f"{symbol}")
    print(f"{'=' * 40}")

    print(f"\nMean:     {stats['mean']:.6f}")
    print(f"Std:      {stats['std']:.6f}")
    print(f"Skew:     {stats['skew']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")

    if hurst:
        print(f"\nFull Hurst: {hurst:.4f}")
        print("Interpretation:", interpret_hurst(hurst))

    print("\nTrend Direction Analysis:")
    if mom20  is not None: print(f"  20-Day Return:  {mom20  * 100:.2f}%")
    if mom50  is not None: print(f"  50-Day Return:  {mom50  * 100:.2f}%")
    if mom100 is not None: print(f"  100-Day Return: {mom100 * 100:.2f}%")

    print(f"\nTrend Classification: {trend_direction}")
    print(f"Trend Strength Score: {trend_score}/3")

    if metrics is not None:
        print("\nRolling Hurst Stats:")
        print(f"  Latest H: {metrics['latest']:.3f}")
        print(f"  Mean H:   {metrics['mean']:.3f}")
        print(f"  Std H:    {metrics['std']:.3f}")

        print("\nRegime Signals:")
        print(f"  Trend Confirmed: {metrics['trend_confirmed']}")
        print(f"  ΔH:              {metrics['delta_h']:.4f}")
        print(f"  % Trending:      {metrics['pct_trending']:.2%}")
        print(f"  Slope:           {metrics['slope']:.6f}")

        print("\nInterpretation:")
        print(interpret_rolling(metrics))

        print("\nCombined Hurst + Momentum Signal:")
        if metrics["trend_confirmed"] and trend_score == 3:
            print("  STRONG LONG STRUCTURE")
            print("  Persistent trend + all momentum horizons bullish.")
        elif metrics["trend_confirmed"] and trend_score == 0:
            print("  STRONG SHORT STRUCTURE")
            print("  Persistent trend + all momentum horizons bearish.")
        elif trend_score >= 2:
            print("  MODERATE BULLISH STRUCTURE")
            print("  Momentum mostly bullish but Hurst confirmation weaker.")
        elif trend_score <= 1:
            print("  MODERATE BEARISH STRUCTURE")
            print("  Momentum mostly bearish but Hurst confirmation weaker.")
        else:
            print("  TRANSITIONAL MARKET")
            print("  Direction and persistence disagree.")

    # HMM OUTPUT
    if hmm_ok and asset.regimes is not None:
        print(f"\n{'=' * 40}")
        print("HMM REGIME DETECTION")
        print(f"{'=' * 40}")

        regime_summary(asset.regimes)

        print("\nState Mean Returns:")
        for label in ["low_vol", "high_vol", "extreme_vol"]:
            m         = state_means[label]
            direction = "↑" if m["mean_return"] > 0 else "↓"
            print(
                f"  {label:12s}: "
                f"return = {m['mean_return']*100:+.3f}%  {direction}   "
                f"vol = {m['mean_vol']*100:.3f}%"
            )

        print("\nTransition Probabilities:")
        for from_state, to_states in trans_matrix.items():
            for to_state, prob in to_states.items():
                print(f"  {from_state:12s} → {to_state:12s}: {prob:.3f}")

        current_regime = asset.regimes[-1]
        print(f"\n{'=' * 40}")
        print("FINAL SIGNAL")
        print(f"{'=' * 40}")
        print(f"  Current Regime : {current_regime.upper()}")
        print(f"  Momentum Score : {trend_score}/3")
        print(f"  Trend Hurst    : {'confirmed' if metrics and metrics['trend_confirmed'] else 'not confirmed'}")
        print()

        low_vol_positive = state_means["low_vol"]["mean_return"] > 0

        if current_regime == "low_vol" and trend_score >= 2 and low_vol_positive:
            print("  ✓ CALM UPTREND")
            print("    Low vol, positive returns, momentum aligned bullish.")
            print("    Best conditions for a long position.")

        elif current_regime == "low_vol" and not low_vol_positive:
            print("  ⚠ CALM DOWNTREND")
            print("    Low vol but returns are negative — organised, steady selling.")
            print("    No panic but price is drifting down. Avoid longs.")

        elif current_regime == "low_vol" and trend_score <= 1:
            print("  ~ CALM BUT DIRECTIONLESS")
            print("    Low vol and momentum is mixed. Wait for clearer signal.")

        elif current_regime == "high_vol" and trend_score <= 1:
            print("  ⚠ VOLATILE DOWNTREND")
            print("    Big swings, momentum bearish. Elevated risk, avoid.")

        elif current_regime == "high_vol" and trend_score >= 2:
            print("  ~ VOLATILE RECOVERY")
            print("    Momentum turning positive but vol is still elevated.")
            print("    Potential entry but wait for vol to calm down first.")

        elif current_regime == "extreme_vol":
            print("  ⚠ CHAOS")
            print("    Extreme moves detected. Stay out or cut size heavily.")
            print("    Not a tradeable environment.")

        else:
            print("  ~ MIXED SIGNAL")
            print("    Regime and momentum are not aligned. No clean trade.")

    # show plot after all output
    plt.show()


if __name__ == "__main__":
    main()
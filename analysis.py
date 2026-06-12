import pandas as pd
import numpy as np
from warehouse import load_from_db


def analyse(symbol: str):
    df = load_from_db(symbol)

    if df is None or df.empty or len(df) < 10:
        print(f"{symbol}: insufficient data")
        return None

    df = df.copy()
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)

    df['returns'] = df['close'].pct_change()
    df.dropna(inplace=True)

    if len(df) < 5:
        print(f"{symbol}: not enough returns")
        return None

    mean = df['returns'].mean()
    std = df['returns'].std()
    kurt = df['returns'].kurtosis()
    skew = df['returns'].skew()

    print(f"\n{symbol.upper()} Statistics")
    print(f"Mean daily return: {mean*100:.3f}%")
    print(f"Std deviation:     {std*100:.3f}%")
    print(f"Kurtosis:          {kurt:.2f}")
    print(f"Skew:              {skew:.2f}")

    # tail classification
    if kurt < 2:
        print("Very light tails")
    elif kurt < 3:
        print("Light tails")
    elif kurt < 5:
        print("Moderate tails")
    elif kurt < 10:
        print("Fat tails")
    else:
        print("Very fat tails")

    # skew classification
    if skew < -1:
        print("Strong negative skew")
    elif skew < -0.5:
        print("Moderate negative skew")
    elif skew < 0.5:
        print("Approximately symmetric")
    elif skew < 1:
        print("Moderate positive skew")
    else:
        print("Strong positive skew")

    return {
        "symbol": symbol,
        "mean": mean,
        "std": std,
        "kurtosis": kurt,
        "skew": skew
    }
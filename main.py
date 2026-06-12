import ccxt
from fetch import fetch_ohlcv
from warehouse import save_to_db, init_db, ticker_exists
from analysis import analyse
from hurst import compute_hurst

# -----------------------------
# INIT DB
# -----------------------------
print("Initializing database...")
init_db()


# -----------------------------
# GET TICKER INPUT
# -----------------------------
raw = input("Enter ticker symbol (e.g. BTC, ETH, SOL): ").strip().upper()

if not raw:
    print("No ticker entered. Exiting.")
    exit()

symbol = raw


# -----------------------------
# FETCH OR LOAD FROM DB
# -----------------------------
print(f"\nChecking database for {symbol}...")

if ticker_exists(symbol):
    print(f"{symbol} found in DB — skipping fetch.")
else:
    print(f"{symbol} not in DB — fetching from Binance...")
    try:
        df = fetch_ohlcv(symbol)
        if df is None or df.empty:
            print(f"No data returned for {symbol}. Exiting.")
            exit()
        save_to_db(df, symbol)
        print(f"{symbol} saved to DB.")
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")
        exit()


# -----------------------------
# ANALYSIS
# -----------------------------
def score(x):
    return (
        x["mean"] * 100
        + 0.4 * x["skew"]
        - 0.1 * x["kurtosis"]
    )

stats = analyse(symbol)
hurst_result = compute_hurst(symbol)

if hurst_result:
    print(f"Hurst Exponent: {hurst_result['hurst']:.3f}")
    print(f"Interpretation: {hurst_result['interpretation']}")

'''if stats:
    stats["score"] = score(stats)
    print(f"\nScore: {round(stats['score'], 3)}")
else:
    print(f"Analysis failed for {symbol}.")'''
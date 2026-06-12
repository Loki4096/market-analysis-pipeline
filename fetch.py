import ccxt
import pandas as pd


def fetch_ohlcv(symbol: str, start_date: str = '2017-01-01') -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    pair = f"{symbol}/USDT"

    since = exchange.parse8601(f'{start_date}T00:00:00Z')
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(pair, '1d', since=since, limit=1000)

        if not ohlcv:
            break

        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1

        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = symbol.upper()

    return df
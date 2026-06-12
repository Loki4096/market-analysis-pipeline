import ccxt


def get_top_symbols(limit=200):
    exchange = ccxt.binance()
    tickers = exchange.fetch_tickers()

    candidates = []

    for symbol, data in tickers.items():
        if symbol.endswith("/USDT"):
            volume = data.get("quoteVolume")

            if volume is not None:
                base = symbol.replace("/USDT", "")
                candidates.append((base, volume))

    candidates.sort(key=lambda x: x[1], reverse=True)

    return [c[0] for c in candidates[:limit]]
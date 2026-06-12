import duckdb
import pandas as pd

DB_PATH = "crypto.db"


def init_db():
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR
        )
    """)

    con.close()


def ticker_exists(symbol: str) -> bool:
    con = duckdb.connect(DB_PATH)

    res = con.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ?",
        [symbol.upper()]
    ).fetchone()

    con.close()
    return res[0] > 0


def save_to_db(df: pd.DataFrame, symbol: str):
    con = duckdb.connect(DB_PATH)

    symbol = symbol.upper()
    df = df.copy()
    df["symbol"] = symbol

    con.execute("DELETE FROM ohlcv WHERE symbol = ?", [symbol])
    con.execute("INSERT INTO ohlcv SELECT * FROM df")

    con.close()


def load_from_db(symbol: str):
    con = duckdb.connect(DB_PATH)

    df = con.execute("""
        SELECT * FROM ohlcv
        WHERE symbol = ?
        ORDER BY timestamp
    """, [symbol.upper()]).df()

    con.close()
    return df
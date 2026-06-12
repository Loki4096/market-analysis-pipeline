import duckdb
import pandas as pd

DB_PATH = ".venv/crypto.db"


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
    """Returns True if any rows exist in the DB for this symbol."""
    con = duckdb.connect(DB_PATH)

    result = con.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ?",
        [symbol.upper()]
    ).fetchone()

    con.close()
    return result[0] > 0


def save_to_db(df: pd.DataFrame, symbol: str):
    con = duckdb.connect(DB_PATH)

    symbol = symbol.upper()
    df = df.copy()
    df["symbol"] = symbol

    # overwrite existing data for this symbol
    con.execute("DELETE FROM ohlcv WHERE symbol = ?", [symbol])
    con.execute("INSERT INTO ohlcv SELECT * FROM df")

    con.close()


def load_from_db(symbol: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)

    df = con.execute("""
        SELECT * FROM ohlcv
        WHERE symbol = ?
        ORDER BY timestamp
    """, [symbol.upper()]).df()

    con.close()
    return df
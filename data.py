from warehouse import load_from_db
import numpy as np


class AssetData:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.df = None
        self.returns = None

    def load(self):
        self.df = load_from_db(self.symbol)

        if self.df is None or len(self.df) < 10:
            return False

        self.df = self.df.drop_duplicates().sort_values("timestamp")
        self.returns = self.df["close"].pct_change().dropna().values

        return True
from warehouse import load_from_db
import numpy as np


class AssetData:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.df = None
        self.returns = None

        # populated by hmm.py
        self.regimes = None   # np.array of state labels per timestep
        self.hmm_model = None   # fitted GaussianHMM object
        self.hmm_features = None  # scaled feature matrix used for fitting

    def load(self):
        self.df = load_from_db(self.symbol)

        if self.df is None or len(self.df) < 10:
            return False

        self.df = self.df.drop_duplicates().sort_values("timestamp")
        self.returns = (
            np.log(self.df["close"] / self.df["close"].shift(1))
            .dropna()
            .values
        )

        return True

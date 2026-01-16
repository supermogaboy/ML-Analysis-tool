import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib


class RegimeClassifier:
    """
    Unsupervised market regime classifier using Gaussian Mixture Models.

    Regimes:
    - Calm
    - General Trend with Risk
    - Volatile Event
    - Crisis
    """

    def __init__(self, n_regimes=4, random_state=42):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state
        )
        self.regime_map = {}

    @staticmethod
    def make_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # Use 'close' if 'Adj Close' not available
        price_col = 'Adj Close' if 'Adj Close' in d.columns else 'close'
        
        d["ret"] = d[price_col].pct_change()
        d["vol_21"] = d["ret"].rolling(21).std()
        d["vol_63"] = d["ret"].rolling(63).std()

        d["trend_63"] = (
            d[price_col] / d[price_col].rolling(63).mean() - 1
        )

        d["drawdown"] = (
            d[price_col] / d[price_col].cummax() - 1
        )

        d["sharpe_63"] = (
            d["ret"].rolling(63).mean() /
            d["ret"].rolling(63).std()
        )

        features = [
            "ret",
            "vol_21",
            "vol_63",
            "trend_63",
            "drawdown",
            "sharpe_63"
        ]

        return d[features].dropna()

    def fit(self, df: pd.DataFrame):
        X = self.make_features(df)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)

        clusters = self.model.predict(X_scaled)
        X["cluster"] = clusters

        summary = X.groupby("cluster").mean()[
            ["vol_63", "trend_63", "drawdown", "sharpe_63"]
        ]

        self.regime_map = self._infer_regimes(summary)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self.make_features(df)
        X_scaled = self.scaler.transform(X)

        clusters = self.model.predict(X_scaled)
        regimes = pd.Series(
            clusters, index=X.index
        ).map(self.regime_map)

        return regimes

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.make_features(df)
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)

        return pd.DataFrame(
            probs,
            index=X.index,
            columns=[f"Regime_{i}" for i in range(self.n_regimes)]
        )

    @staticmethod
    def _infer_regimes(summary: pd.DataFrame) -> dict:
        regime_map = {}

        for c, row in summary.iterrows():
            if row["vol_63"] < 0.012 and row["drawdown"] > -0.10:
                regime_map[c] = "Calm"
            elif row["vol_63"] > 0.03 and row["drawdown"] < -0.25:
                regime_map[c] = "Crisis"
            elif row["vol_63"] > 0.025:
                regime_map[c] = "Volatile Event"
            else:
                regime_map[c] = "General Trend with Risk"

        return regime_map

    def save(self, path: str):
        joblib.dump(
            {
                "scaler": self.scaler,
                "model": self.model,
                "regime_map": self.regime_map,
            },
            path
        )

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.model = obj["model"]
        self.regime_map = obj["regime_map"]

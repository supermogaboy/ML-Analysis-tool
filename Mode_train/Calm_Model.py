"""
Calm Regime Specialist
======================

This module implements an explainable, conservative specialist model for the
Calm market regime on QQQ daily data.

Key design choices
------------------
- **Directional model**: Multinomial Logistic Regression (P(down), P(flat), P(up))
- **Magnitude models**: Elastic Net regression for:
    - Expected upside magnitude U  (conditional on positive forward return)
    - Expected downside magnitude D (absolute value, conditional on negative return)
- **Features (all from past data only)**:
    - 1d, 5d, 10d log returns
    - 10‑day realized volatility of 1d log returns
    - 14‑day RSI
    - 20‑day Bollinger Band Z‑score (based on close)
    - Short‑term trend proxy: (EMA20 − EMA50) / close
    - 21‑day drawdown (as a stress / calm filter)
- **Labels** (using Config.HORIZON_DAYS and Config.FLAT_BAND):
    - forward_return: log(adj_close[t + H] / adj_close[t])
    - class:
        - down: forward_return < -epsilon
        - flat: |forward_return| <= epsilon
        - up  : forward_return > +epsilon
    - U = max(forward_return, 0)
    - D = max(-forward_return, 0)
- **Calm regime filter**:
    - Low recent realized volatility
    - Mild drawdowns (no large stress events)
    - Implemented as quantile‑based thresholds on:
        - 10‑day realized volatility of 1d log returns
        - 21‑day drawdown

The module trains three sklearn pipelines:
    - Directional: StandardScaler + LogisticRegression (multinomial)
    - Upside:      StandardScaler + ElasticNet
    - Downside:    StandardScaler + ElasticNet

Models and metadata are saved to disk for downstream use, including Kelly
position sizing modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Config import HORIZON_DAYS, FLAT_BAND, RANDOM_SEED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CalmRegimeConfig:
    """
    Configuration for the Calm Regime specialist.

    Attributes
    ----------
    horizon_days : int
        Prediction horizon used for labels (forward returns).
    flat_band : float
        Threshold for classifying "flat" returns (in log‑return units).
    vol_quantile : float
        Upper quantile for realized volatility to qualify as "calm".
    dd_quantile : float
        Lower quantile for drawdown to qualify as "calm"
        (drawdown is typically negative; higher == less stressed).
    min_calm_samples : int
        Minimum number of calm samples required to train models.
    test_size_fraction : float
        Fraction of calm samples reserved for out‑of‑sample evaluation.
    random_state : int
        Seed for reproducibility where applicable.
    """

    horizon_days: int = HORIZON_DAYS
    flat_band: float = FLAT_BAND
    vol_quantile: float = 0.4
    dd_quantile: float = 0.4
    min_calm_samples: int = 500
    test_size_fraction: float = 0.2
    random_state: int = RANDOM_SEED


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interpretable, leakage‑free features from daily QQQ data.

    Expected input columns
    ----------------------
    - 'date'
    - 'adj_close'
    - 'close'

    Additional OHLCV columns (open, high, low, volume) are ignored here but
    preserved in the original DataFrame if needed elsewhere.
    """
    # Ensure sorted by time
    df = df.sort_values("date").reset_index(drop=True)

    price = df["adj_close"].astype(float)
    close = df["close"].astype(float)

    # 1d log return
    log_ret_1d = np.log(price / price.shift(1))

    # 5d and 10d log returns (sums of 1d log returns)
    log_ret_5d = log_ret_1d.rolling(window=5).sum()
    log_ret_10d = log_ret_1d.rolling(window=10).sum()

    # 10‑day realized volatility of 1d log returns
    vol_10d = log_ret_1d.rolling(window=10).std()

    # RSI(14) on close
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing
    roll_period = 14
    avg_gain = gain.ewm(alpha=1.0 / roll_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / roll_period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Band Z‑score (20‑day) on close
    bb_window = 20
    ma_20 = close.rolling(window=bb_window).mean()
    std_20 = close.rolling(window=bb_window).std()
    bb_z = (close - ma_20) / std_20

    # Trend proxy: (EMA20 − EMA50) / close
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    trend_short = (ema_20 - ema_50) / close

    # 21‑day drawdown: price / rolling_max(21) − 1
    roll_max_21 = price.rolling(window=21).max()
    drawdown_21 = price / roll_max_21 - 1.0

    features = pd.DataFrame(
        {
            "log_ret_1d": log_ret_1d,
            "log_ret_5d": log_ret_5d,
            "log_ret_10d": log_ret_10d,
            "vol_10d": vol_10d,
            "rsi_14": rsi_14,
            "bb_z_20": bb_z,
            "trend_ema20_50": trend_short,
            "drawdown_21": drawdown_21,
        },
        index=df.index,
    )

    features["date"] = df["date"].values
    return features


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------


def compute_labels(
    df: pd.DataFrame,
    horizon_days: int,
    flat_band: float,
) -> pd.DataFrame:
    """
    Compute forward returns and classification / regression targets.

    Parameters
    ----------
    df : DataFrame
        Must contain 'date' and 'adj_close' columns.
    horizon_days : int
        Forward horizon in trading days.
    flat_band : float
        Band for classifying "flat" in terms of log‑returns.

    Returns
    -------
    DataFrame
        Columns:
            - 'date'
            - 'forward_return'
            - 'class_label'  (0=down, 1=flat, 2=up)
            - 'upside'       (U)
            - 'downside'     (D, absolute value)
    """
    df = df.sort_values("date").reset_index(drop=True)
    price = df["adj_close"].astype(float)

    fwd_price = price.shift(-horizon_days)
    forward_return = np.log(fwd_price / price)

    # Classification labels
    down_mask = forward_return < -flat_band
    up_mask = forward_return > flat_band
    flat_mask = (~down_mask) & (~up_mask)

    class_label = pd.Series(np.nan, index=df.index, dtype=float)
    class_label[down_mask] = 0.0
    class_label[flat_mask] = 1.0
    class_label[up_mask] = 2.0

    # Magnitude labels
    upside = forward_return.clip(lower=0.0)
    downside = (-forward_return).clip(lower=0.0)

    labels = pd.DataFrame(
        {
            "date": df["date"].values,
            "forward_return": forward_return,
            "class_label": class_label,
            "upside": upside,
            "downside": downside,
        },
        index=df.index,
    )

    return labels


# ---------------------------------------------------------------------------
# Calm regime filter
# ---------------------------------------------------------------------------


def identify_calm_regime(
    features: pd.DataFrame,
    config: CalmRegimeConfig,
) -> pd.Series:
    """
    Identify calm regime observations based on volatility and drawdown.

    Rules (quantile‑based, purely cross‑sectional within the sample):
    - realized volatility (vol_10d) below vol_quantile
    - drawdown_21 above dd_quantile (less negative drawdown)
    """
    vol = features["vol_10d"]
    dd = features["drawdown_21"]

    vol_threshold = vol.quantile(config.vol_quantile)
    dd_threshold = dd.quantile(config.dd_quantile)

    calm_mask = (vol <= vol_threshold) & (dd >= dd_threshold)
    return calm_mask.fillna(False)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def build_calm_dataset(
    raw_df: pd.DataFrame,
    config: CalmRegimeConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build feature and label arrays restricted to the Calm regime.

    Parameters
    ----------
    raw_df : DataFrame
        Raw OHLCV data loaded from CSV.
    config : CalmRegimeConfig
        Configuration object.

    Returns
    -------
    X_calm : DataFrame
        Feature matrix for calm regime samples.
    y_class_calm : Series
        Class labels (0, 1, 2) for calm regime samples.
    y_up_calm : Series
        Upside magnitude labels (U) for calm regime samples.
    y_down_calm : Series
        Downside magnitude labels (D) for calm regime samples.
    """
    features = compute_features(raw_df)
    labels = compute_labels(
        raw_df,
        horizon_days=config.horizon_days,
        flat_band=config.flat_band,
    )

    # Align by date and drop rows with any NaN in required columns
    merged = features.merge(labels, on="date", how="inner", suffixes=("_feat", "_lbl"))
    merged = merged.dropna(
        subset=[
            "log_ret_1d",
            "log_ret_5d",
            "log_ret_10d",
            "vol_10d",
            "rsi_14",
            "bb_z_20",
            "trend_ema20_50",
            "drawdown_21",
            "class_label",
            "upside",
            "downside",
        ]
    )

    # Identify calm regime
    calm_mask = identify_calm_regime(
        features=merged[
            [
                "date",
                "log_ret_1d",
                "log_ret_5d",
                "log_ret_10d",
                "vol_10d",
                "rsi_14",
                "bb_z_20",
                "trend_ema20_50",
                "drawdown_21",
            ]
        ].set_index(merged.index),
        config=config,
    )

    calm_data = merged[calm_mask]

    if len(calm_data) < config.min_calm_samples:
        raise ValueError(
            f"Insufficient calm samples ({len(calm_data)}) "
            f"for training; minimum required is {config.min_calm_samples}."
        )

    feature_cols = [
        "log_ret_1d",
        "log_ret_5d",
        "log_ret_10d",
        "vol_10d",
        "rsi_14",
        "bb_z_20",
        "trend_ema20_50",
        "drawdown_21",
    ]

    X_calm = calm_data[feature_cols].copy()
    y_class_calm = calm_data["class_label"].astype(int)
    y_up_calm = calm_data["upside"].astype(float)
    y_down_calm = calm_data["downside"].astype(float)

    return X_calm, y_class_calm, y_up_calm, y_down_calm


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def time_series_train_test_split(
    n_samples: int,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronological train/test split for time‑series data.
    """
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction should be between 0 and 0.5 for stability.")

    split_idx = int(n_samples * (1.0 - test_fraction))
    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, n_samples)
    return train_idx, test_idx


def build_directional_pipeline(
    random_state: int,
) -> Pipeline:
    """
    Build the multinomial logistic regression pipeline.
    """
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=0.5,  # strong L2 regularization for stability
        max_iter=1000,
        n_jobs=None,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", clf),
        ]
    )
    return pipe


def build_magnitude_pipeline() -> Pipeline:
    """
    Build the Elastic Net regression pipeline for magnitude modeling.
    """
    reg = ElasticNet(
        alpha=0.01,  # aggressive but not extreme regularization
        l1_ratio=0.5,
        max_iter=1000,
        fit_intercept=True,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("elastic_net", reg),
        ]
    )
    return pipe


def train_calm_specialist(
    X_calm: pd.DataFrame,
    y_class_calm: pd.Series,
    y_up_calm: pd.Series,
    y_down_calm: pd.Series,
    config: CalmRegimeConfig,
) -> Tuple[Pipeline, Pipeline, Pipeline, Dict[str, float]]:
    """
    Train Calm regime models with time‑series‑aware evaluation.

    Returns
    -------
    directional_model : Pipeline
        Fitted multinomial logistic regression pipeline.
    upside_model : Pipeline
        Fitted Elastic Net pipeline for upside magnitude.
    downside_model : Pipeline
        Fitted Elastic Net pipeline for downside magnitude.
    metrics : dict
        Simple out‑of‑sample evaluation metrics on the hold‑out calm set.
    """
    n_samples = len(X_calm)
    train_idx, test_idx = time_series_train_test_split(
        n_samples=n_samples,
        test_fraction=config.test_size_fraction,
    )

    X_train, X_test = X_calm.iloc[train_idx], X_calm.iloc[test_idx]
    y_class_train, y_class_test = y_class_calm.iloc[train_idx], y_class_calm.iloc[test_idx]
    y_up_train, y_up_test = y_up_calm.iloc[train_idx], y_up_calm.iloc[test_idx]
    y_down_train, y_down_test = y_down_calm.iloc[train_idx], y_down_calm.iloc[test_idx]

    # Directional model
    directional_model = build_directional_pipeline(random_state=config.random_state)
    directional_model.fit(X_train, y_class_train)
    prob_test = directional_model.predict_proba(X_test)
    log_loss_test = log_loss(y_class_test, prob_test, labels=[0, 1, 2])

    # Magnitude models (note: we fit on all samples; regressors learn conditional magnitudes)
    upside_model = build_magnitude_pipeline()
    downside_model = build_magnitude_pipeline()

    upside_model.fit(X_train, y_up_train)
    downside_model.fit(X_train, y_down_train)

    up_pred = upside_model.predict(X_test)
    down_pred = downside_model.predict(X_test)

    metrics = {
        "directional_log_loss": float(log_loss_test),
        "upside_mae": float(mean_absolute_error(y_up_test, up_pred)),
        "upside_rmse": float(np.sqrt(mean_squared_error(y_up_test, up_pred))),
        "downside_mae": float(mean_absolute_error(y_down_test, down_pred)),
        "downside_rmse": float(np.sqrt(mean_squared_error(y_down_test, down_pred))),
        "n_calm_train": int(len(train_idx)),
        "n_calm_test": int(len(test_idx)),
    }

    # Refit models on ALL calm data for production use
    directional_model.fit(X_calm, y_class_calm)
    upside_model.fit(X_calm, y_up_calm)
    downside_model.fit(X_calm, y_down_calm)

    return directional_model, upside_model, downside_model, metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calm_models(
    directional_model: Pipeline,
    upside_model: Pipeline,
    downside_model: Pipeline,
    feature_names: Tuple[str, ...],
    metrics: Dict[str, float],
    output_dir: str = "Model",
) -> None:
    """
    Save trained models and metadata to disk.

    Files written
    ------------
    - calm_directional.pkl : multinomial logistic regression pipeline
    - calm_upside.pkl      : Elastic Net upside pipeline
    - calm_downside.pkl    : Elastic Net downside pipeline
    - calm_metadata.json   : feature names, metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    directional_path = os.path.join(output_dir, "calm_directional.pkl")
    upside_path = os.path.join(output_dir, "calm_upside.pkl")
    downside_path = os.path.join(output_dir, "calm_downside.pkl")
    metadata_path = os.path.join(output_dir, "calm_metadata.json")

    joblib.dump(directional_model, directional_path)
    joblib.dump(upside_model, upside_path)
    joblib.dump(downside_model, downside_path)

    metadata = {
        "feature_names": list(feature_names),
        "metrics": metrics,
    }
    pd.Series(metadata).to_json(metadata_path)


# ---------------------------------------------------------------------------
# Top‑level training script
# ---------------------------------------------------------------------------


def load_qqq_daily_csv(path: str) -> pd.DataFrame:
    """
    Load QQQ daily CSV as produced by MakeCSV.py.

    Expected columns (case‑sensitive):
    - date, adj_close, close, high, low, open, volume
    """
    df = pd.read_csv(path)
    # Normalise column names to lower snake_case if needed
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    return df


def train_calm_regime_specialist(
    csv_path: str = "data/raw/QQQ_1d.csv",
    output_dir: str = "Model",
    config: CalmRegimeConfig | None = None,
) -> Dict[str, float]:
    """
    End‑to‑end training pipeline for the Calm Regime specialist.

    Steps
    -----
    1. Load daily QQQ CSV.
    2. Compute interpretable features.
    3. Build labels with horizon and flat band from Config.
    4. Filter to calm regime observations only.
    5. Train:
        - multinomial logistic regression for direction
        - Elastic Net for upside magnitude
        - Elastic Net for downside magnitude
    6. Evaluate on a chronological hold‑out subset of calm data.
    7. Refit on all calm data and persist models + metadata to disk.

    Returns
    -------
    dict
        Metrics from the hold‑out calm evaluation set.
    """
    if config is None:
        config = CalmRegimeConfig()

    raw_df = load_qqq_daily_csv(csv_path)

    X_calm, y_class_calm, y_up_calm, y_down_calm = build_calm_dataset(
        raw_df=raw_df,
        config=config,
    )

    directional_model, upside_model, downside_model, metrics = train_calm_specialist(
        X_calm=X_calm,
        y_class_calm=y_class_calm,
        y_up_calm=y_up_calm,
        y_down_calm=y_down_calm,
        config=config,
    )

    save_calm_models(
        directional_model=directional_model,
        upside_model=upside_model,
        downside_model=downside_model,
        feature_names=tuple(X_calm.columns),
        metrics=metrics,
        output_dir=output_dir,
    )

    return metrics


if __name__ == "__main__":
    # Simple CLI entry point for training the Calm specialist.
    metrics_out = train_calm_regime_specialist()
    print("Calm Regime specialist trained.")
    print("Hold‑out calm metrics:")
    for k, v in metrics_out.items():
        print(f"  {k}: {v}")


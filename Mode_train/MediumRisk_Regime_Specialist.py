"""
Medium Risk Regime Specialist
=============================

This module implements a specialist model for the Medium Risk market regime on QQQ daily data.

Key design choices
------------------
- **Directional model**: GBDT multiclass classifier (XGBoost or sklearn HistGradientBoosting)
  with optional probability calibration (P(down), P(flat), P(up))
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
- **Medium Risk regime filter**:
    - NOT calm AND NOT crisis
    - Calm: vol_10d <= quantile(calm_vol_q) AND drawdown_21 >= quantile(calm_dd_q)
    - Crisis: vol_10d >= quantile(crisis_vol_q) OR drawdown_21 <= quantile(crisis_dd_q)
    - Medium Risk: NOT calm AND NOT crisis

The module trains three sklearn pipelines:
    - Directional: GBDT multiclass classifier (with optional calibration)
    - Upside:      StandardScaler + ElasticNet
    - Downside:    StandardScaler + ElasticNet

Models and metadata are saved to disk for downstream use, including Kelly
position sizing modules.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure src directory is on path for Config import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Config import HORIZON_DAYS, FLAT_BAND, RANDOM_SEED

# Try to import XGBoost, fallback to sklearn HistGradientBoosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# Try to import calibration
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MediumRiskRegimeConfig:
    """
    Configuration for the Medium Risk Regime specialist.

    Attributes
    ----------
    horizon_days : int
        Prediction horizon used for labels (forward returns).
    flat_band : float
        Threshold for classifying "flat" returns (in log‑return units).
    calm_vol_q : float
        Upper quantile for realized volatility to qualify as "calm".
    calm_dd_q : float
        Lower quantile for drawdown to qualify as "calm"
        (drawdown is typically negative; higher == less stressed).
    crisis_vol_q : float
        Lower quantile for realized volatility to qualify as "crisis".
    crisis_dd_q : float
        Upper quantile for drawdown to qualify as "crisis"
        (drawdown is typically negative; lower == more stressed).
    min_medium_samples : int
        Minimum number of medium-risk samples required to train models.
    test_size_fraction : float
        Fraction of medium-risk samples reserved for out‑of‑sample evaluation.
    random_state : int
        Seed for reproducibility where applicable.
    enable_calibration : bool
        Whether to apply probability calibration to the directional model.
    """

    horizon_days: int = HORIZON_DAYS
    flat_band: float = FLAT_BAND
    calm_vol_q: float = 0.35
    calm_dd_q: float = 0.45
    crisis_vol_q: float = 0.80
    crisis_dd_q: float = 0.15
    min_medium_samples: int = 800
    test_size_fraction: float = 0.2
    random_state: int = RANDOM_SEED
    enable_calibration: bool = True


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
# Medium Risk regime filter
# ---------------------------------------------------------------------------


def identify_medium_risk_regime(
    features: pd.DataFrame,
    config: MediumRiskRegimeConfig,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Identify medium risk regime observations based on volatility and drawdown.

    Rules (quantile‑based, computed on merged dataset after dropping NaNs):
    - Calm: vol_10d <= quantile(calm_vol_q) AND drawdown_21 >= quantile(calm_dd_q)
    - Crisis: vol_10d >= quantile(crisis_vol_q) OR drawdown_21 <= quantile(crisis_dd_q)
    - Medium Risk: NOT calm AND NOT crisis

    Returns
    -------
    medium_risk_mask : Series
        Boolean mask indicating medium risk samples.
    thresholds : dict
        Dictionary with actual threshold values used (vol and drawdown cutoffs).
    """
    vol = features["vol_10d"]
    dd = features["drawdown_21"]

    # Compute quantile thresholds
    vol_calm_threshold = vol.quantile(config.calm_vol_q)
    dd_calm_threshold = dd.quantile(config.calm_dd_q)
    vol_crisis_threshold = vol.quantile(config.crisis_vol_q)
    dd_crisis_threshold = dd.quantile(config.crisis_dd_q)

    # Define calm and crisis masks
    calm_mask = (vol <= vol_calm_threshold) & (dd >= dd_calm_threshold)
    crisis_mask = (vol >= vol_crisis_threshold) | (dd <= dd_crisis_threshold)

    # Medium risk is NOT calm AND NOT crisis
    medium_risk_mask = (~calm_mask) & (~crisis_mask)
    medium_risk_mask = medium_risk_mask.fillna(False)

    thresholds = {
        "vol_calm_threshold": float(vol_calm_threshold),
        "dd_calm_threshold": float(dd_calm_threshold),
        "vol_crisis_threshold": float(vol_crisis_threshold),
        "dd_crisis_threshold": float(dd_crisis_threshold),
    }

    return medium_risk_mask, thresholds


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def build_medium_risk_dataset(
    raw_df: pd.DataFrame,
    config: MediumRiskRegimeConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, float]]:
    """
    Build feature and label arrays restricted to the Medium Risk regime.

    Parameters
    ----------
    raw_df : DataFrame
        Raw OHLCV data loaded from CSV.
    config : MediumRiskRegimeConfig
        Configuration object.

    Returns
    -------
    X_medium : DataFrame
        Feature matrix for medium-risk regime samples.
    y_class_medium : Series
        Class labels (0, 1, 2) for medium-risk regime samples.
    y_up_medium : Series
        Upside magnitude labels (U) for medium-risk regime samples.
    y_down_medium : Series
        Downside magnitude labels (D) for medium-risk regime samples.
    thresholds : dict
        Dictionary with threshold values used for regime identification.
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

    # Identify medium risk regime (compute thresholds on merged dataset)
    medium_risk_mask, thresholds = identify_medium_risk_regime(
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

    medium_data = merged[medium_risk_mask]

    if len(medium_data) < config.min_medium_samples:
        raise ValueError(
            f"Insufficient medium-risk samples ({len(medium_data)}) "
            f"for training; minimum required is {config.min_medium_samples}."
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

    X_medium = medium_data[feature_cols].copy()
    y_class_medium = medium_data["class_label"].astype(int)
    y_up_medium = medium_data["upside"].astype(float)
    y_down_medium = medium_data["downside"].astype(float)

    return X_medium, y_class_medium, y_up_medium, y_down_medium, thresholds


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
) -> Tuple[Pipeline, str]:
    """
    Build the GBDT multiclass classifier pipeline.

    Uses XGBoost if available, otherwise falls back to sklearn HistGradientBoostingClassifier.

    Returns
    -------
    Pipeline
        The GBDT classifier pipeline with StandardScaler.
    str
        Model backend name ("xgboost" or "sklearn_hgb").
    """
    if XGBOOST_AVAILABLE:
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            max_depth=3,
            learning_rate=0.08,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        model_backend = "xgboost"
    else:
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=3,
            learning_rate=0.08,
            max_iter=500,
            random_state=random_state,
        )
        model_backend = "sklearn_hgb"

    # Note: GBDT doesn't strictly need scaling, but we keep it for consistency
    # with the Calm model structure
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gbdt", clf),
        ]
    )
    return pipe, model_backend


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


def train_medium_risk_specialist(
    X_medium: pd.DataFrame,
    y_class_medium: pd.Series,
    y_up_medium: pd.Series,
    y_down_medium: pd.Series,
    config: MediumRiskRegimeConfig,
) -> Tuple[object, Pipeline, Pipeline, Dict[str, float], str, Dict[str, Any]]:
    """
    Train Medium Risk regime models with time‑series‑aware evaluation.

    Returns
    -------
    directional_model : Pipeline or CalibratedClassifierCV
        Fitted GBDT multiclass classifier pipeline (optionally calibrated).
    upside_model : Pipeline
        Fitted Elastic Net pipeline for upside magnitude.
    downside_model : Pipeline
        Fitted Elastic Net pipeline for downside magnitude.
    metrics : dict
        Out‑of‑sample evaluation metrics on the hold‑out medium-risk set.
    model_backend : str
        Backend used ("xgboost" or "sklearn_hgb").
    calibration_info : dict
        Information about calibration (enabled, method).
    """
    n_samples = len(X_medium)
    train_idx, test_idx = time_series_train_test_split(
        n_samples=n_samples,
        test_fraction=config.test_size_fraction,
    )

    X_train, X_test = X_medium.iloc[train_idx], X_medium.iloc[test_idx]
    y_class_train, y_class_test = y_class_medium.iloc[train_idx], y_class_medium.iloc[test_idx]
    y_up_train, y_up_test = y_up_medium.iloc[train_idx], y_up_medium.iloc[test_idx]
    y_down_train, y_down_test = y_down_medium.iloc[train_idx], y_down_medium.iloc[test_idx]

    # Build base directional model
    base_pipeline, model_backend = build_directional_pipeline(random_state=config.random_state)

    # Calibration setup (chronological split within training set)
    calibration_info = {"enabled": False, "method": None}
    if config.enable_calibration and CALIBRATION_AVAILABLE:
        # Split training into subtrain (80%) and calib (20%)
        n_train = len(X_train)
        calib_split_idx = int(n_train * 0.8)
        subtrain_idx = np.arange(0, calib_split_idx)
        calib_idx = np.arange(calib_split_idx, n_train)

        X_subtrain = X_train.iloc[subtrain_idx]
        y_subtrain = y_class_train.iloc[subtrain_idx]
        X_calib = X_train.iloc[calib_idx]
        y_calib = y_class_train.iloc[calib_idx]

        # Fit base model on subtrain
        base_pipeline.fit(X_subtrain, y_subtrain)

        # Determine calibration method based on per-class counts
        class_counts = y_calib.value_counts().reindex([0, 1, 2], fill_value=0)
        min_class_count = class_counts.min()
        if min_class_count >= 50:
            calib_method = "isotonic"
        else:
            calib_method = "sigmoid"

        # Calibrate on calib set
        calibrated_model = CalibratedClassifierCV(
            base_pipeline,
            method=calib_method,
            cv="prefit",
        )
        calibrated_model.fit(X_calib, y_calib)

        calibration_info = {"enabled": True, "method": calib_method}
        directional_model = calibrated_model
    else:
        # No calibration: fit on full training set
        base_pipeline.fit(X_train, y_class_train)
        directional_model = base_pipeline

    # Evaluate directional model
    prob_test = directional_model.predict_proba(X_test)
    log_loss_test = log_loss(y_class_test, prob_test, labels=[0, 1, 2])
    y_pred_test = directional_model.predict(X_test)
    accuracy_test = accuracy_score(y_class_test, y_pred_test)
    f1_macro_test = f1_score(y_class_test, y_pred_test, average="macro", labels=[0, 1, 2])
    cm = confusion_matrix(y_class_test, y_pred_test, labels=[0, 1, 2])

    # Magnitude models (note: we fit on all samples; regressors learn conditional magnitudes)
    upside_model = build_magnitude_pipeline()
    downside_model = build_magnitude_pipeline()

    upside_model.fit(X_train, y_up_train)
    downside_model.fit(X_train, y_down_train)

    up_pred = upside_model.predict(X_test)
    down_pred = downside_model.predict(X_test)

    metrics = {
        "directional_log_loss": float(log_loss_test),
        "directional_accuracy": float(accuracy_test),
        "directional_macro_f1": float(f1_macro_test),
        "upside_mae": float(mean_absolute_error(y_up_test, up_pred)),
        "upside_rmse": float(np.sqrt(mean_squared_error(y_up_test, up_pred))),
        "downside_mae": float(mean_absolute_error(y_down_test, down_pred)),
        "downside_rmse": float(np.sqrt(mean_squared_error(y_down_test, down_pred))),
        "n_medium_train": int(len(train_idx)),
        "n_medium_test": int(len(test_idx)),
    }

    # Store confusion matrix as nested list for JSON serialization
    metrics["confusion_matrix"] = cm.tolist()

    # Refit models on ALL medium-risk data for production use
    if config.enable_calibration and CALIBRATION_AVAILABLE:
        # Split chronologically: base_fit (90%) and calib_fit (10%)
        n_final = len(X_medium)
        base_fit_idx = int(n_final * 0.9)
        base_fit_indices = np.arange(0, base_fit_idx)
        calib_fit_indices = np.arange(base_fit_idx, n_final)

        X_base_fit = X_medium.iloc[base_fit_indices]
        y_base_fit = y_class_medium.iloc[base_fit_indices]
        X_calib_fit = X_medium.iloc[calib_fit_indices]
        y_calib_fit = y_class_medium.iloc[calib_fit_indices]

        # Fit base model ONLY on base_fit (no leakage)
        base_pipeline_final, _ = build_directional_pipeline(random_state=config.random_state)
        base_pipeline_final.fit(X_base_fit, y_base_fit)

        # Calibrate ONLY on calib_fit using cv="prefit"
        calib_method = calibration_info.get("method", "sigmoid")
        if calib_method is None:
            calib_method = "sigmoid"
        calibrated_model_final = CalibratedClassifierCV(
            base_pipeline_final,
            method=calib_method,
            cv="prefit",
        )
        calibrated_model_final.fit(X_calib_fit, y_calib_fit)
        directional_model = calibrated_model_final
    else:
        base_pipeline_final, _ = build_directional_pipeline(random_state=config.random_state)
        base_pipeline_final.fit(X_medium, y_class_medium)
        directional_model = base_pipeline_final

    upside_model.fit(X_medium, y_up_medium)
    downside_model.fit(X_medium, y_down_medium)

    return directional_model, upside_model, downside_model, metrics, model_backend, calibration_info


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_medium_risk_models(
    directional_model: object,
    upside_model: Pipeline,
    downside_model: Pipeline,
    feature_names: Tuple[str, ...],
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    model_backend: str,
    calibration_info: Dict[str, Any],
    config: MediumRiskRegimeConfig,
    output_dir: str = "Model",
) -> None:
    """
    Save trained models and metadata to disk.

    Files written
    ------------
    - mediumrisk_directional.pkl : GBDT multiclass classifier pipeline
    - mediumrisk_upside.pkl      : Elastic Net upside pipeline
    - mediumrisk_downside.pkl    : Elastic Net downside pipeline
    - mediumrisk_metadata.json   : feature names, metrics, thresholds, config
    """
    os.makedirs(output_dir, exist_ok=True)

    directional_path = os.path.join(output_dir, "mediumrisk_directional.pkl")
    upside_path = os.path.join(output_dir, "mediumrisk_upside.pkl")
    downside_path = os.path.join(output_dir, "mediumrisk_downside.pkl")
    metadata_path = os.path.join(output_dir, "mediumrisk_metadata.json")

    joblib.dump(directional_model, directional_path)
    joblib.dump(upside_model, upside_path)
    joblib.dump(downside_model, downside_path)

    metadata = {
        "feature_names": list(feature_names),
        "metrics": metrics,
        "thresholds": thresholds,
        "model_backend": model_backend,
        "calibration_info": calibration_info,
        "config": {
            "horizon_days": config.horizon_days,
            "flat_band": config.flat_band,
            "calm_vol_q": config.calm_vol_q,
            "calm_dd_q": config.calm_dd_q,
            "crisis_vol_q": config.crisis_vol_q,
            "crisis_dd_q": config.crisis_dd_q,
            "min_medium_samples": config.min_medium_samples,
            "test_size_fraction": config.test_size_fraction,
            "random_state": config.random_state,
            "enable_calibration": config.enable_calibration,
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


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


def train_medium_risk_regime_specialist(
    csv_path: str = "data/raw/QQQ_1d.csv",
    output_dir: str = "Model",
    config: MediumRiskRegimeConfig | None = None,
) -> Dict[str, float]:
    """
    End‑to‑end training pipeline for the Medium Risk Regime specialist.

    Steps
    -----
    1. Load daily QQQ CSV.
    2. Compute interpretable features.
    3. Build labels with horizon and flat band from Config.
    4. Filter to medium-risk regime observations only.
    5. Train:
        - GBDT multiclass classifier for direction (with optional calibration)
        - Elastic Net for upside magnitude
        - Elastic Net for downside magnitude
    6. Evaluate on a chronological hold‑out subset of medium-risk data.
    7. Refit on all medium-risk data and persist models + metadata to disk.

    Returns
    -------
    dict
        Metrics from the hold‑out medium-risk evaluation set.
    """
    if config is None:
        config = MediumRiskRegimeConfig()

    raw_df = load_qqq_daily_csv(csv_path)

    X_medium, y_class_medium, y_up_medium, y_down_medium, thresholds = build_medium_risk_dataset(
        raw_df=raw_df,
        config=config,
    )

    directional_model, upside_model, downside_model, metrics, model_backend, calibration_info = train_medium_risk_specialist(
        X_medium=X_medium,
        y_class_medium=y_class_medium,
        y_up_medium=y_up_medium,
        y_down_medium=y_down_medium,
        config=config,
    )

    save_medium_risk_models(
        directional_model=directional_model,
        upside_model=upside_model,
        downside_model=downside_model,
        feature_names=tuple(X_medium.columns),
        metrics=metrics,
        thresholds=thresholds,
        model_backend=model_backend,
        calibration_info=calibration_info,
        config=config,
        output_dir=output_dir,
    )

    return metrics


if __name__ == "__main__":
    # Simple CLI entry point for training the Medium Risk specialist.
    metrics_out = train_medium_risk_regime_specialist()
    print("Medium Risk specialist trained.")
    print("Hold‑out medium-risk metrics:")
    for k, v in metrics_out.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix:\n{np.array(metrics_out['confusion_matrix'])}")

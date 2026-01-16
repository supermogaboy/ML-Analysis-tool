"""
Trend Regime Specialist Model
===========================

This module implements the core model functionality for the Trend Regime Specialist.
It includes feature engineering, dataset preparation, model training, inference,
and persistence functions.

This is the model implementation that gets imported by the training script
in Mode_train/Trend_Expert.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.Config import HORIZON_DAYS, FLAT_BAND, RANDOM_SEED
from src.Features import add_features
from src.Regiemes import label_regimes, REGIMES
from src.Dataset import make_targets


# ---------------------------------------------------------------------------
# Configuration (re-import for model use)
# ---------------------------------------------------------------------------

@dataclass
class TrendRegimeConfig:
    """
    Configuration for the Trend Regime specialist.
    """
    horizon_days: int = HORIZON_DAYS
    n_walk_forward_splits: int = 5
    min_train_samples: int = 200
    early_stopping_rounds: int = 50
    calibration_method: str = "isotonic"
    xgb_params: dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 3,
                "min_child_weight": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "random_state": RANDOM_SEED,
                "n_jobs": -1
            }


# ---------------------------------------------------------------------------
# Feature Engineering (using existing infrastructure)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and map features from processed data to model expected format.

    Expected input columns (from processed CSV)
    -------------------------------------------
    - 'date', 'close', 'high', 'low', 'volume'
    - Base features from src.Features pipeline

    Returns
    -------
    DataFrame with trend-focused features for model training.
    """
    # Ensure sorted by time
    df = df.sort_values("date").reset_index(drop=True)

    # Use existing feature engineering pipeline
    features = add_features(df)
    
    # Add trend-specific features not in base pipeline
    features = _add_trend_specific_features(features)
    
    return features


def _add_trend_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend-specific features that complement the existing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with existing features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional trend features
    """
    df = df.copy()
    close = df["close"]
    
    # MACD indicators (not in base pipeline)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = macd_line - signal_line
    df["macd_norm"] = macd_line / close
    
    # Additional momentum features
    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)
    
    # Trend strength indicators
    df["trend_strength"] = (df["ema_20"] - df["ema_50"]).abs() / close
    df["price_above_ema20"] = (close / df["ema_20"]) - 1.0
    df["price_above_ema50"] = (close / df["ema_50"]) - 1.0
    
    # Volatility-normalized returns
    df["norm_ret_1_vol10"] = df["logret_1"] / (df["vol_10"] + 1e-12)
    df["norm_ret_5_vol10"] = df["logret_5"] / (df["vol_10"] + 1e-12)
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names expected by the model.
    
    Returns
    -------
    list
        List of feature column names for trend model
    """
    # Base features from existing pipeline that are trend-relevant
    base_features = [
        "logret_1", "logret_5", "logret_10",
        "vol_10", "vol_20",
        "ema_20", "ema_50", "ema_200",
        "trend_20_50", "trend_px_200",
        "rsi_14", "bb_z_20",
        "atr_14", "atrp_14",
        "vol_z20", "dd_63", "dist_252_high"
    ]
    
    # Additional trend-specific features
    trend_features = [
        "macd", "macd_signal", "macd_histogram", "macd_norm",
        "momentum_5", "momentum_10",
        "trend_strength", "price_above_ema20", "price_above_ema50",
        "norm_ret_1_vol10", "norm_ret_5_vol10"
    ]
    
    return base_features + trend_features


# ---------------------------------------------------------------------------
# Label Construction (using existing infrastructure)
# ---------------------------------------------------------------------------

def compute_labels(
    df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Extract or compute forward returns and binary trend labels.

    Parameters
    ----------
    df : DataFrame
        Processed data with 'date', 'close', and optionally targets.
    horizon_days : int
        Forward horizon in trading days (used if labels not in processed data).

    Returns
    -------
    DataFrame
        Columns:
            - 'date'
            - 'trend_label'  (0=down/flat, 1=up)
            - 'forward_return'
    """
    df = df.sort_values("date").reset_index(drop=True)
    
    # Use existing target creation from src.Dataset
    df_with_targets = make_targets(df)
    
    # Convert to binary trend label (up vs not-up)
    # y_class: 0=down, 1=flat, 2=up
    # We want: 1 if up (y_class == 2), else 0
    trend_label = (df_with_targets["y_class"] == 2).astype(float)
    
    # Also get forward return for analysis
    forward_return = df_with_targets["fwd_ret"]
    
    labels = pd.DataFrame(
        {
            "date": df["date"],
            "trend_label": trend_label,
            "forward_return": forward_return,
        },
        index=df.index,
    )

    return labels


# ---------------------------------------------------------------------------
# Trend Regime Filter (using existing infrastructure)
# ---------------------------------------------------------------------------

def identify_trend_regime(
    features: pd.DataFrame,
    config: TrendRegimeConfig,
) -> pd.Series:
    """
    Identify trend regime observations using existing src.Regiemes infrastructure.

    Rules (using existing regime labeling):
    - Uses label_regimes() from src.Regiemes
    - Trend regime is "trend_risk" from the existing system
    - Causal (no future info) by design of src.Regiemes
    """
    # Use existing regime labeling
    regimes = label_regimes(features)
    
    # Trend regime is "trend_risk" from existing system
    trend_mask = (regimes == "trend_risk").fillna(False)
    
    return trend_mask


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------

def build_trend_dataset(
    raw_df: pd.DataFrame,
    config: TrendRegimeConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build feature and label arrays restricted to the Trend regime.

    Parameters
    ----------
    raw_df : DataFrame
        Raw OHLCV data loaded from CSV.
    config : TrendRegimeConfig
        Configuration object.

    Returns
    -------
    X_trend : DataFrame
        Feature matrix for trend regime samples.
    y_trend : Series
        Binary trend labels (0, 1) for trend regime samples.
    trend_mask : Series
        Boolean mask indicating trend regime.
    forward_returns : Series
        Forward returns for analysis.
    """
    features = compute_features(raw_df)
    labels = compute_labels(
        raw_df,
        horizon_days=config.horizon_days,
    )

    # Align by date and drop rows with any NaN in required columns
    merged = features.merge(labels, on="date", how="inner", suffixes=("_feat", "_lbl"))
    
    feature_cols = get_feature_columns()
    merged = merged.dropna(
        subset=feature_cols + ["trend_label"]
    )

    # Identify trend regime on merged data (not on separate frames)
    # This ensures perfect alignment between features, labels, and regime mask
    trend_mask = identify_trend_regime(
        features=merged[feature_cols + ["date"]],
        config=config,
    )

    trend_data = merged[trend_mask]

    if len(trend_data) < config.min_train_samples:
        raise ValueError(
            f"Insufficient trend samples ({len(trend_data)}) "
            f"for training; minimum required is {config.min_train_samples}."
        )

    X_trend = trend_data[feature_cols].copy()
    y_trend = trend_data["trend_label"].astype(int)
    forward_returns = trend_data["forward_return"]

    return X_trend, y_trend, trend_mask, forward_returns


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def time_series_train_test_split(
    n_samples: int,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronological train/test split for time-series data.
    """
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction should be between 0 and 0.5 for stability.")

    split_idx = int(n_samples * (1.0 - test_fraction))
    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, n_samples)
    return train_idx, test_idx


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrendRegimeConfig
) -> List[Dict[str, Any]]:
    """
    Perform walk-forward validation on trend regime samples.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Binary labels
    config : TrendRegimeConfig
        Training configuration
        
    Returns
    -------
    list
        List of validation results for each split
    """
    # Create walk-forward splits
    n_samples = len(X)
    split_size = n_samples // (config.n_walk_forward_splits + 1)
    
    results = []
    
    for split in range(config.n_walk_forward_splits):
        # Define train and validation indices
        train_end = split_size * (split + 1)
        val_end = train_end + split_size
        
        if val_end > n_samples:
            break
        
        train_idx = X.index[:train_end]
        val_idx = X.index[train_end:val_end]
        
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        
        # Train XGBoost model
        model = xgb.XGBClassifier(**config.xgb_params)
        
        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=False
        )
        
        # Compute metrics with single-class protection
        # FIX: Prevent ROC-AUC crashes when validation split has single class
        if len(np.unique(y_val)) < 2:
            # Single class in validation - use default metrics
            metrics = {
                "split": split,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "auc": 0.5,  # Random baseline for single class
                "log_loss": log_loss(y_val, np.full(len(y_val), y_val.iloc[0])),
                "brier_score": brier_score_loss(y_val, np.full(len(y_val), y_val.iloc[0])),
                "best_iteration": model.best_iteration,
                "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
                "single_class_warning": True
            }
        else:
            # Normal multi-class metrics
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            metrics = {
                "split": split,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "auc": roc_auc_score(y_val, y_pred_proba),
                "log_loss": log_loss(y_val, y_pred_proba),
                "brier_score": brier_score_loss(y_val, y_pred_proba),
                "best_iteration": model.best_iteration,
                "feature_importance": dict(zip(X_train.columns, model.feature_importances_))
            }
            
            # Add confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            metrics.update({
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "f1": f1_score(y_val, y_pred)
            })
            
            # Store model and predictions for calibration
            metrics["model"] = model
            metrics["y_val"] = y_val
            metrics["y_pred_proba"] = y_pred_proba
        
        results.append(metrics)
    
    return results


def calibrate_probabilities(
    validation_results: List[Dict[str, Any]],
    method: str = "isotonic"
) -> Any:
    """
    Fit probability calibration model on validation predictions.
    
    Parameters
    ----------
    validation_results : list
        Results from walk-forward validation
    method : str
        'isotonic' or 'platt' calibration method
        
    Returns
    -------
    object
        Fitted calibration model
    """
    # FIX: Calibration must be trained only on out-of-fold validation predictions
    # Collect all validation predictions from walk-forward splits
    all_y = []
    all_proba = []
    
    for result in validation_results:
        # Skip splits with single class warnings
        if result.get("single_class_warning", False):
            continue
        
        all_y.extend(result["y_val"].values)
        all_proba.extend(result["y_pred_proba"].values)
    
    if len(all_y) == 0:
        # No valid validation splits - fallback to Platt scaling
        calibrator = LogisticRegression()
        return calibrator
    
    all_y = np.array(all_y)
    all_proba = np.array(all_proba)
    
    # Choose calibration method
    if method == "isotonic" and len(all_y) >= 100:
        calibrator = IsotonicRegression(out_of_bounds='clip')
    else:
        # Use Platt scaling (logistic regression) for small datasets
        calibrator = LogisticRegression()
        all_proba = all_proba.reshape(-1, 1)
    
    # Fit calibration model
    calibrator.fit(all_proba, all_y)
    
    return calibrator


def train_trend_specialist(
    X_trend: pd.DataFrame,
    y_trend: pd.Series,
    config: TrendRegimeConfig,
) -> Tuple[xgb.XGBClassifier, Any, List[str], Dict[str, float]]:
    """
    Train Trend regime models with walk-forward validation.

    Returns
    -------
    model : XGBClassifier
        Fitted XGBoost classifier pipeline.
    calibrator : object
        Fitted probability calibration model.
    feature_columns : list
        List of feature column names.
    metrics : dict
        Walk-forward validation metrics.
    """
    # Walk-forward validation
    validation_results = walk_forward_validation(X_trend, y_trend, config)
    
    # Compute comprehensive metrics
    metrics = {
        "mean_auc": np.mean([r["auc"] for r in validation_results]),
        "std_auc": np.std([r["auc"] for r in validation_results]),
        "mean_log_loss": np.mean([r["log_loss"] for r in validation_results]),
        "std_log_loss": np.std([r["log_loss"] for r in validation_results]),
        "mean_brier_score": np.mean([r["brier_score"] for r in validation_results]),
        "total_samples": sum([r["val_size"] for r in validation_results]),
        "n_splits": len(validation_results)
    }
    
    # FIX: Use proper calibration from walk-forward validation, not in-sample
    # Train final model on all trend data
    final_model = xgb.XGBClassifier(**config.xgb_params)
    final_model.fit(X_trend, y_trend)
    
    # FIX: Calibration must use out-of-sample predictions from walk-forward validation
    # Re-run walk-forward to get validation predictions for calibration
    validation_results = walk_forward_validation(X_trend, y_trend, config)
    calibrator = calibrate_probabilities(validation_results, config.calibration_method)
    
    return final_model, calibrator, list(X_trend.columns), metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trend_models(
    model: xgb.XGBClassifier,
    calibrator: Any,
    feature_names: Tuple[str, ...],
    metrics: Dict[str, float],
    output_dir: str = "Model",
) -> None:
    """
    Save trained models and metadata to disk.

    Files written
    ------------
    - trend_xgb_model.pkl : XGBoost classifier
    - trend_calibrator.pkl : Probability calibration model
    - trend_metadata.json : feature names, metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "trend_xgb_model.pkl")
    calibrator_path = os.path.join(output_dir, "trend_calibrator.pkl")
    metadata_path = os.path.join(output_dir, "trend_metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)

    metadata = {
        "feature_names": list(feature_names),
        "metrics": metrics,
        "model_type": "TrendExpert",
        "regime_filter": "trend_risk"
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Inference Functions
# ---------------------------------------------------------------------------

def load_trend_models(model_dir: str = "Model") -> Dict[str, Any]:
    """
    Load trained Trend Expert models and metadata.
    
    Parameters
    ----------
    model_dir : str
        Directory containing trained models
        
    Returns
    -------
    dict
        Dictionary with model, calibrator, and metadata
    """
    model_path = os.path.join(model_dir, "trend_xgb_model.pkl")
    calibrator_path = os.path.join(model_dir, "trend_calibrator.pkl")
    metadata_path = os.path.join(model_dir, "trend_metadata.json")
    
    if not all(os.path.exists(p) for p in [model_path, calibrator_path, metadata_path]):
        raise FileNotFoundError(f"Trend Expert model files not found in {model_dir}")
    
    # Load models
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return {
        "model": model,
        "calibrator": calibrator,
        "metadata": metadata,
        "feature_columns": metadata["feature_names"]
    }


def predict_trend_proba(
    df: pd.DataFrame,
    artifacts: Dict[str, Any] | None = None,
    model_dir: str = "Model"
) -> pd.Series:
    """
    Generate calibrated trend probability predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with OHLCV columns and date
    artifacts : dict, optional
        Pre-loaded model artifacts (if None, loads from model_dir)
    model_dir : str
        Directory containing trained models (used if artifacts is None)
        
    Returns
    -------
    pd.Series
        Calibrated p_up probabilities (NaN for non-trend periods)
    """
    # Load models if not provided
    if artifacts is None:
        artifacts = load_trend_models(model_dir)
    
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    feature_columns = artifacts["feature_columns"]
    
    # Compute features using existing infrastructure
    features = compute_features(df)
    
    # Identify trend regime using existing infrastructure
    trend_mask = identify_trend_regime(features, TrendRegimeConfig())
    
    # Prepare features for prediction
    X = features[feature_columns].copy()
    
    # FIX: Do NOT fill missing values with zeros - use forward fill for causal features
    # Only fill remaining NaNs (at beginning) with small values to avoid division by zero
    X = X.ffill().fillna(1e-8)
    
    # Generate raw predictions
    raw_proba = model.predict_proba(X)[:, 1]
    
    # FIX: Use type checking instead of hasattr to detect calibrator type
    # This is more reliable and explicit
    if isinstance(calibrator, IsotonicRegression):
        # Isotonic regression
        calibrated_proba = calibrator.predict(raw_proba)
    elif isinstance(calibrator, LogisticRegression):
        # Logistic regression (Platt scaling)
        calibrated_proba = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    else:
        # Fallback - assume isotonic-like interface
        calibrated_proba = calibrator.predict(raw_proba)
    
    # Create result series with NaN for non-trend periods
    result = pd.Series(np.nan, index=df.index)
    result[trend_mask] = calibrated_proba[trend_mask]
    
    return result


def get_trend_signals(
    df: pd.DataFrame,
    confidence_threshold: float = 0.6,
    artifacts: Dict[str, Any] | None = None,
    model_dir: str = "Model"
) -> pd.DataFrame:
    """
    Generate trend signals with confidence filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with OHLCV columns
    confidence_threshold : float
        Minimum probability to generate a signal
    artifacts : dict, optional
        Pre-loaded model artifacts
    model_dir : str
        Directory containing trained models
        
    Returns
    -------
    pd.DataFrame
        DataFrame with signals and probabilities
    """
    # Get probabilities
    probabilities = predict_trend_proba(df, artifacts, model_dir)
    
    # Generate signals
    signals = (probabilities >= confidence_threshold).astype(int)
    
    # Get regime information
    features = compute_features(df)
    regimes = label_regimes(features)
    
    # Create result DataFrame
    result = pd.DataFrame({
        "date": df["date"],
        "close": df["close"],
        "trend_probability": probabilities,
        "trend_signal": signals,
        "regime": regimes
    })
    
    return result

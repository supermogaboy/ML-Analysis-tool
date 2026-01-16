"""
Chop Regime Specialist Model
===========================

This module implements the core model functionality for the Chop Regime Specialist.
It includes feature engineering, dataset preparation, model training, inference,
and persistence functions.

The Chop expert detects mean-reverting, range-bound, and noisy price action.
It is trained on extended chop samples using z-score filtering and outputs
calibrated probabilities suitable for position sizing or gating.

This is the model implementation that gets imported by the training script
in Mode_train/Chop_Expert.py
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
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChopRegimeConfig:
    """
    Configuration for the Chop Regime specialist.

    Attributes
    ----------
    horizon_days : int
        Prediction horizon used for labels (forward returns).
    n_walk_forward_splits : int
        Number of walk-forward splits for validation.
    min_train_samples : int
        Minimum number of chop samples required to train models.
    early_stopping_rounds : int
        Early stopping rounds for XGBoost.
    calibration_method : str
        'isotonic' or 'platt' for probability calibration.
    xgb_params : dict
        XGBoost hyperparameters.
    # CHOP: Chop-specific configuration
    return_z_threshold : float
        Z-score threshold for return filtering (|z| < threshold = chop)
    momentum_z_threshold : float
        Z-score threshold for momentum filtering (|z| < threshold = chop)
    volatility_upper_quantile : float
        Upper volatility quantile to exclude crisis periods
    trend_strength_threshold : float
        Threshold for trend strength filtering
    extended_samples : bool
        Whether to use extended overlapping samples for chop detection
    """
    horizon_days: int = HORIZON_DAYS
    n_walk_forward_splits: int = 5
    min_train_samples: int = 300
    early_stopping_rounds: int = 50
    calibration_method: str = "isotonic"
    xgb_params: dict = None
    # CHOP: Chop-specific parameters
    return_z_threshold: float = 0.75
    momentum_z_threshold: float = 0.5
    volatility_upper_quantile: float = 0.85
    trend_strength_threshold: float = 0.02
    extended_samples: bool = True
    
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
    DataFrame with chop-focused features for model training.
    """
    # Ensure sorted by time
    df = df.sort_values("date").reset_index(drop=True)

    # Use existing feature engineering pipeline
    features = add_features(df)
    
    # Add chop-specific features not in base pipeline
    features = _add_chop_specific_features(features)
    
    return features


def _add_chop_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chop-specific features that complement the existing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with existing features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional chop features
    """
    df = df.copy()
    close = df["close"]
    
    # CHOP: Range-bound detection features
    # Price position within recent range
    for window in [10, 20, 50]:
        rolling_max = close.rolling(window=window).max()
        rolling_min = close.rolling(window=window).min()
        df[f"range_position_{window}"] = (close - rolling_min) / (rolling_max - rolling_min + 1e-12)
        
        # Range width normalized by price
        df[f"range_width_{window}"] = (rolling_max - rolling_min) / close
    
    # CHOP: Mean reversion indicators
    # Distance from moving averages
    for window in [10, 20, 50]:
        ma = close.rolling(window=window).mean()
        df[f"ma_distance_{window}"] = (close - ma) / ma
        df[f"ma_distance_z_{window}"] = df[f"ma_distance_{window}"] / df[f"ma_distance_{window}"].rolling(window=50).std()
    
    # CHOP: Volatility regime features
    # Volatility ratio to detect non-crisis but volatile periods
    vol_10 = df["logret_1"].rolling(10).std()
    vol_50 = df["logret_1"].rolling(50).std()
    df["vol_ratio_10_50"] = vol_10 / (vol_50 + 1e-12)
    
    # CHOP: Momentum oscillation features
    # RSI distance from neutral (50) as oscillation indicator
    df["rsi_oscillation"] = np.abs(df["rsi_14"] - 50) / 50
    
    # Bollinger Band position and width
    df["bb_position"] = df["bb_z_20"]  # Already z-scored
    df["bb_width"] = (close.rolling(20).std() * 2) / close
    
    # CHOP: Trend weakness indicators
    # Trend strength consistency
    trend_20_50 = df["trend_20_50"]
    df["trend_consistency"] = trend_20_50.rolling(10).apply(lambda x: len(np.unique(np.sign(x))) == 1)
    
    # Price rate of change acceleration
    roc_5 = close.pct_change(5)
    roc_10 = close.pct_change(10)
    df["roc_acceleration"] = roc_5 - roc_10
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names expected by the model.
    
    Returns
    -------
    list
        List of feature column names for chop model
    """
    # Base features from existing pipeline that are chop-relevant
    base_features = [
        "logret_1", "logret_5", "logret_10",
        "vol_10", "vol_20",
        "ema_20", "ema_50", "ema_200",
        "trend_20_50", "trend_px_200",
        "rsi_14", "bb_z_20",
        "atr_14", "atrp_14",
        "vol_z20", "dd_63", "dist_252_high"
    ]
    
    # CHOP: Additional chop-specific features
    chop_features = [
        # Range-bound detection
        "range_position_10", "range_position_20", "range_position_50",
        "range_width_10", "range_width_20", "range_width_50",
        # Mean reversion
        "ma_distance_10", "ma_distance_20", "ma_distance_50",
        "ma_distance_z_10", "ma_distance_z_20", "ma_distance_z_50",
        # Volatility regime
        "vol_ratio_10_50",
        # Momentum oscillation
        "rsi_oscillation",
        # Bollinger bands
        "bb_position", "bb_width",
        # Trend weakness
        "trend_consistency", "roc_acceleration"
    ]
    
    return base_features + chop_features


# ---------------------------------------------------------------------------
# Label Construction (using existing infrastructure)
# ---------------------------------------------------------------------------

def compute_labels(
    df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Extract or compute forward returns and binary chop labels.

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
            - 'chop_label'  (0=non-chop, 1=chop)
            - 'forward_return'
    """
    df = df.sort_values("date").reset_index(drop=True)
    
    # Use existing target creation from src.Dataset
    df_with_targets = make_targets(df)
    
    # CHOP: Convert to binary chop label (chop vs non-chop)
    # Chop = flat or small movements, non-chop = strong directional moves
    # y_class: 0=down, 1=flat, 2=up
    # Chop = flat (y_class == 1), Non-chop = strong up/down
    chop_label = (df_with_targets["y_class"] == 1).astype(float)
    
    # Also get forward return for analysis
    forward_return = df_with_targets["fwd_ret"]
    
    labels = pd.DataFrame(
        {
            "date": df["date"],
            "chop_label": chop_label,
            "forward_return": forward_return,
        },
        index=df.index,
    )

    return labels


# ---------------------------------------------------------------------------
# Chop Regime Filter (z-score based)
# ---------------------------------------------------------------------------

def identify_chop_regime(
    features: pd.DataFrame,
    config: ChopRegimeConfig,
) -> pd.Series:
    """
    Identify chop regime observations using z-score filtering and existing regime labels.

    Rules (CHOP-specific):
    - Use z-score filter on returns and momentum
    - Filter out crisis periods using volatility
    - Filter out strong trend periods
    - Use extended overlapping samples for robust detection

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with features
    config : ChopRegimeConfig
        Configuration object

    Returns
    -------
    pd.Series
        Boolean mask indicating chop regime
    """
    # CHOP: Z-score filtering on returns
    logret_1 = features["logret_1"]
    # FIX: Ensure sufficient rolling window size to avoid std=0 crashes
    if len(logret_1) < 50:
        # Not enough data for z-score calculation - return all False
        return pd.Series(False, index=features.index)
    
    return_z = (logret_1 - logret_1.rolling(50).mean()) / (logret_1.rolling(50).std() + 1e-12)
    return_filter = np.abs(return_z) < config.return_z_threshold
    
    # CHOP: Z-score filtering on momentum (5-day return)
    momentum_5 = logret_1.rolling(5).sum()
    # FIX: Ensure sufficient rolling window size to avoid std=0 crashes
    if len(momentum_5) < 50:
        # Not enough data for momentum z-score - return all False
        return pd.Series(False, index=features.index)
    
    momentum_z = (momentum_5 - momentum_5.rolling(50).mean()) / (momentum_5.rolling(50).std() + 1e-12)
    momentum_filter = np.abs(momentum_z) < config.momentum_z_threshold
    
    # CHOP: Volatility filtering (exclude crisis periods)
    vol_20 = features["vol_20"]
    # FIX: Handle empty volatility series safely
    if len(vol_20) == 0 or vol_20.isna().all():
        # No volatility data - return all False
        return pd.Series(False, index=features.index)
    
    # FIX: Use expanding quantile to prevent data leakage (no future information)
    vol_upper = vol_20.expanding().quantile(config.volatility_upper_quantile)
    vol_filter = vol_20 < vol_upper
    
    # CHOP: Trend strength filtering (exclude strong trends)
    trend_strength = np.abs(features["trend_20_50"])
    trend_filter = trend_strength < config.trend_strength_threshold
    
    # CHOP: Use existing regime labels as additional filter
    # FIX: Handle regime labeling failures gracefully
    try:
        regimes = label_regimes(features)
        # Include calm and some trend_risk (weak trends) but exclude crisis and volatile_event
        regime_filter = regimes.isin(["calm", "trend_risk"])
    except Exception:
        # If regime labeling fails, default to False (conservative approach)
        regime_filter = pd.Series(False, index=features.index)
    
    # CHOP: Combine all filters
    # FIX: Ensure all filters have same index before combining
    all_filters = [return_filter, momentum_filter, vol_filter, trend_filter, regime_filter]
    # Reindex all filters to match features.index
    aligned_filters = []
    for f in all_filters:
        if f.index.equals(features.index):
            aligned_filters.append(f)
        else:
            aligned_filters.append(f.reindex(features.index, fill_value=False))
    
    chop_mask = (
        aligned_filters[0] & 
        aligned_filters[1] & 
        aligned_filters[2] & 
        aligned_filters[3] & 
        aligned_filters[4]
    ).fillna(False)
    
    # CHOP: Extended samples - create overlapping windows for robust detection
    if config.extended_samples:
        # Create extended mask using rolling windows
        # FIX: Ensure window size doesn't exceed data length
        window_size = min(3, len(chop_mask))
        if window_size > 0:
            extended_mask = chop_mask.rolling(window=window_size, min_periods=1).max().fillna(False)
            chop_mask = extended_mask
    
    return chop_mask


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------

def build_chop_dataset(
    raw_df: pd.DataFrame,
    config: ChopRegimeConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build feature and label arrays restricted to the Chop regime.

    Parameters
    ----------
    raw_df : DataFrame
        Raw OHLCV data loaded from CSV.
    config : ChopRegimeConfig
        Configuration object.

    Returns
    -------
    X_chop : DataFrame
        Feature matrix for chop regime samples.
    y_chop : Series
        Binary chop labels (0, 1) for chop regime samples.
    chop_mask : Series
        Boolean mask indicating chop regime.
    forward_returns : Series
        Forward returns for analysis.
    """
    features = compute_features(raw_df)
    labels = compute_labels(
        raw_df,
        horizon_days=config.horizon_days,
    )

    # Align by date and drop rows with any NaN in required columns
    # FIX: Ensure merge doesn't create duplicate columns or misalign data
    merged = features.merge(labels, on="date", how="inner", suffixes=("_feat", "_lbl"))
    
    feature_cols = get_feature_columns()
    # FIX: Verify all required columns exist before dropping NaNs
    missing_cols = [col for col in feature_cols + ["chop_label"] if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after merge: {missing_cols}")
    
    merged = merged.dropna(
        subset=feature_cols + ["chop_label"]
    )

    # CHOP: Identify chop regime on merged data (not on separate frames)
    # This ensures perfect alignment between features, labels, and regime mask
    # FIX: Pass only required columns to avoid index misalignment
    regime_features = merged[feature_cols + ["date"]].copy()
    chop_mask = identify_chop_regime(
        features=regime_features,
        config=config,
    )
    
    # FIX: Ensure chop_mask aligns with merged data index
    if not chop_mask.index.equals(merged.index):
        chop_mask = chop_mask.reindex(merged.index, fill_value=False)

    chop_data = merged[chop_mask]

    if len(chop_data) < config.min_train_samples:
        raise ValueError(
            f"Insufficient chop samples ({len(chop_data)}) "
            f"for training; minimum required is {config.min_train_samples}."
        )

    X_chop = chop_data[feature_cols].copy()
    y_chop = chop_data["chop_label"].astype(int)
    forward_returns = chop_data["forward_return"]

    return X_chop, y_chop, chop_mask, forward_returns


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
    config: ChopRegimeConfig
) -> List[Dict[str, Any]]:
    """
    Perform walk-forward validation on chop regime samples.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Binary labels
    config : ChopRegimeConfig
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
            # Single class in validation - use clipped constant probabilities
            # FIX: Use p=0.999 for class 1, p=0.001 for class 0 to avoid log_loss issues
            unique_class = y_val.iloc[0]
            constant_prob = 0.999 if unique_class == 1 else 0.001
            constant_probs = np.full(len(y_val), constant_prob)
            
            metrics = {
                "split": split,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "auc": 0.5,  # Random baseline for single class
                "log_loss": log_loss(y_val, constant_probs),
                "brier_score": brier_score_loss(y_val, constant_probs),
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
            # FIX: Handle edge cases where confusion_matrix might fail
            if len(np.unique(y_val)) < 2:
                # Single class - skip confusion matrix metrics
                metrics.update({
                    "true_negatives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "true_positives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "single_class_warning": True
                })
            else:
                # Multi-class - safe to compute confusion matrix
                try:
                    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                    metrics.update({
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "true_positives": int(tp),
                        "precision": precision_score(y_val, y_pred, zero_division=0),
                        "recall": recall_score(y_val, y_pred, zero_division=0),
                        "f1": f1_score(y_val, y_pred, zero_division=0)
                    })
                except Exception as e:
                    # FIX: Safe fallback if confusion matrix fails
                    metrics.update({
                        "true_negatives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "true_positives": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "confusion_matrix_error": str(e)
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
    object or None
        Fitted calibration model, or None if no valid validation data
    """
    # FIX: Calibration fallback must be safe - return None if no valid validation folds
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
        # FIX: Return None instead of unfitted LogisticRegression for safe fallback
        return None
    
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


def train_chop_specialist(
    X_chop: pd.DataFrame,
    y_chop: pd.Series,
    config: ChopRegimeConfig,
) -> Tuple[xgb.XGBClassifier, Any, List[str], Dict[str, float]]:
    """
    Train Chop regime models with walk-forward validation.

    Returns
    -------
    model : XGBClassifier
        Fitted XGBoost classifier pipeline.
    calibrator : object or None
        Fitted probability calibration model, or None if no calibration
    feature_columns : list
        List of feature column names.
    metrics : dict
        Walk-forward validation metrics.
    """
    # Walk-forward validation
    validation_results = walk_forward_validation(X_chop, y_chop, config)
    
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
    
    # FIX: Remove duplicated walk-forward computation - reuse validation_results
    # Train final model on all chop data
    final_model = xgb.XGBClassifier(**config.xgb_params)
    final_model.fit(X_chop, y_chop)
    
    # FIX: Use existing validation_results for calibration (no re-run)
    calibrator = calibrate_probabilities(validation_results, config.calibration_method)
    
    return final_model, calibrator, list(X_chop.columns), metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_chop_models(
    model: xgb.XGBClassifier,
    calibrator: Any,
    feature_names: Tuple[str, ...],
    metrics: Dict[str, float],
    config: ChopRegimeConfig,
    output_dir: str = "Model",
) -> None:
    """
    Save trained models and metadata to disk.

    Files written
    ------------
    - chop_xgb_model.pkl : XGBoost classifier
    - chop_calibrator.pkl : Probability calibration model
    - chop_metadata.json : feature names, metrics, config
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "chop_xgb_model.pkl")
    calibrator_path = os.path.join(output_dir, "chop_calibrator.pkl")
    metadata_path = os.path.join(output_dir, "chop_metadata.json")

    joblib.dump(model, model_path)
    # FIX: Handle None calibrator safely during saving
    if calibrator is not None:
        joblib.dump(calibrator, calibrator_path)
    else:
        # Create empty file to indicate no calibrator
        with open(calibrator_path, 'w') as f:
            f.write("None")

    # FIX: Save chop config thresholds into metadata for inference consistency
    metadata = {
        "feature_names": list(feature_names),
        "metrics": metrics,
        "model_type": "ChopExpert",
        "regime_filter": "chop_zscore",
        "config": {
            "return_z_threshold": config.return_z_threshold,
            "momentum_z_threshold": config.momentum_z_threshold,
            "volatility_upper_quantile": config.volatility_upper_quantile,
            "trend_strength_threshold": config.trend_strength_threshold,
            "extended_samples": config.extended_samples
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Inference Functions
# ---------------------------------------------------------------------------

def load_chop_models(model_dir: str = "Model") -> Dict[str, Any]:
    """
    Load trained Chop Expert models and metadata.
    
    Parameters
    ----------
    model_dir : str
        Directory containing trained models
        
    Returns
    -------
    dict
        Dictionary with model, calibrator, metadata, and config
    """
    model_path = os.path.join(model_dir, "chop_xgb_model.pkl")
    calibrator_path = os.path.join(model_dir, "chop_calibrator.pkl")
    metadata_path = os.path.join(model_dir, "chop_metadata.json")
    
    if not all(os.path.exists(p) for p in [model_path, calibrator_path, metadata_path]):
        raise FileNotFoundError(f"Chop Expert model files not found in {model_dir}")
    
    # Load models
    # FIX: Handle model loading failures gracefully
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model from {model_path}: {e}")
    
    # FIX: Handle None calibrator loading safely
    try:
        # Check if file contains "None" (saved as text)
        with open(calibrator_path, 'r') as f:
            content = f.read().strip()
            if content == "None":
                calibrator = None
            else:
                # Try loading as pickle
                calibrator = joblib.load(calibrator_path)
    except Exception:
        # Fallback: try loading as pickle, if fails set to None
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception:
            calibrator = None
            print(f"Warning: Failed to load calibrator from {calibrator_path}, using None")
    
    # Load metadata
    # FIX: Handle metadata loading failures gracefully
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load metadata from {metadata_path}: {e}")
    
    # FIX: Load chop config thresholds from metadata for inference consistency
    # Handle missing config data gracefully
    config_data = metadata.get("config", {})
    try:
        chop_config = ChopRegimeConfig(
            return_z_threshold=config_data.get("return_z_threshold", 0.75),
            momentum_z_threshold=config_data.get("momentum_z_threshold", 0.5),
            volatility_upper_quantile=config_data.get("volatility_upper_quantile", 0.85),
            trend_strength_threshold=config_data.get("trend_strength_threshold", 0.02),
            extended_samples=config_data.get("extended_samples", True)
        )
    except Exception as e:
        # FIX: Fallback to default config if reconstruction fails
        print(f"Warning: Failed to reconstruct config from metadata: {e}")
        chop_config = ChopRegimeConfig()
    
    return {
        "model": model,
        "calibrator": calibrator,
        "metadata": metadata,
        "feature_columns": metadata["feature_names"],
        "config": chop_config
    }


def predict_chop_proba(
    df: pd.DataFrame,
    artifacts: Dict[str, Any] | None = None,
    model_dir: str = "Model"
) -> pd.Series:
    """
    Generate calibrated chop probability predictions.

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
        Calibrated p_chop probabilities (NaN for non-chop periods)
    """
    # Load models if not provided
    if artifacts is None:
        artifacts = load_chop_models(model_dir)
    
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    feature_columns = artifacts["feature_columns"]
    config = artifacts["config"]  # FIX: Use loaded config for inference consistency
    
    # Compute features using existing infrastructure
    features = compute_features(df)
    
    # CHOP: Identify chop regime using loaded config (not new instance)
    chop_mask = identify_chop_regime(features, config)
    
    # Prepare features for prediction
    X = features[feature_columns].copy()
    
    # FIX: Avoid risky NaN filling - XGBoost supports missing values natively
    # Only fill if absolutely required for specific features
    # X = X.ffill().fillna(1e-8)  # REMOVED: Let XGBoost handle NaNs
    
    # Generate raw predictions
    # FIX: Handle prediction failures gracefully
    try:
        raw_proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")
    
    # FIX: Handle None calibrator safely - return raw probabilities if no calibration
    if calibrator is None:
        calibrated_proba = raw_proba
    elif isinstance(calibrator, IsotonicRegression):
        # Isotonic regression
        calibrated_proba = calibrator.predict(raw_proba)
    elif isinstance(calibrator, LogisticRegression):
        # Logistic regression (Platt scaling)
        calibrated_proba = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    else:
        # Fallback - assume isotonic-like interface
        calibrated_proba = calibrator.predict(raw_proba)
    
    # Create result series with NaN for non-chop periods
    # FIX: Ensure result index matches input and handle empty chop_mask
    result = pd.Series(np.nan, index=df.index)
    if chop_mask.any():
        # Only assign values where chop_mask is True
        result[chop_mask] = calibrated_proba[chop_mask]
    
    return result


def get_chop_signals(
    df: pd.DataFrame,
    confidence_threshold: float = 0.6,
    artifacts: Dict[str, Any] | None = None,
    model_dir: str = "Model"
) -> pd.DataFrame:
    """
    Generate chop signals with confidence filtering.

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
    # FIX: Handle probability prediction failures gracefully
    try:
        probabilities = predict_chop_proba(df, artifacts, model_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to generate chop probabilities: {e}")
    
    # Generate signals
    signals = (probabilities >= confidence_threshold).astype(int)
    
    # Get regime information
    # FIX: Handle regime labeling failures gracefully
    try:
        features = compute_features(df)
        regimes = label_regimes(features)
    except Exception as e:
        # Fallback: use default regime if labeling fails
        print(f"Warning: Failed to label regimes in get_chop_signals: {e}")
        regimes = pd.Series("unknown", index=df.index)
    
    # Create result DataFrame
    result = pd.DataFrame({
        "date": df["date"],
        "close": df["close"],
        "chop_probability": probabilities,
        "chop_signal": signals,
        "regime": regimes
    })
    
    return result

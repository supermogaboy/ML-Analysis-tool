"""
Crisis Regime Specialist Model
============================

This module implements the core model functionality for the Crisis Regime Specialist.
It uses a HAR-RV (Heterogeneous Autoregressive Realized Volatility) model
to forecast near-term market volatility and applies conservative risk-off rules
when predicted volatility is abnormally high.

The Crisis expert is purely defensive - it controls exposure rather than
predicting direction, prioritizing capital preservation over opportunity.

This is the model implementation that gets imported by the training script
in Mode_train/Crisis_Expert.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.Config import HORIZON_DAYS, FLAT_BAND, RANDOM_SEED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrisisRegimeConfig:
    """
    Configuration for the Crisis Regime specialist.

    Attributes
    ----------
    volatility_window : int
        Window for daily realized volatility calculation.
    weekly_window : int
        Window for weekly average realized volatility.
    monthly_window : int
        Window for monthly average realized volatility.
    forecast_horizon : int
        Days ahead to forecast volatility.
    n_walk_forward_splits : int
        Number of walk-forward splits for validation.
    min_train_samples : int
        Minimum number of samples required to train models.
    ridge_alpha : float
        Regularization parameter for Ridge regression.
    crisis_quantile : float
        Quantile for crisis detection (e.g., 0.90 = 90th percentile).
    crisis_lookback : int
        Lookback window for crisis threshold calculation.
    min_exposure : float
        Minimum exposure level during crisis (0.0 = fully defensive).
    normal_exposure : float
        Normal exposure level outside crisis (1.0 = full exposure).
    """
    volatility_window: int = 1
    weekly_window: int = 5
    monthly_window: int = 22
    forecast_horizon: int = HORIZON_DAYS
    n_walk_forward_splits: int = 5
    min_train_samples: int = 500
    ridge_alpha: float = 1.0
    crisis_quantile: float = 0.90
    crisis_lookback: int = 252  # ~1 year of trading days
    min_exposure: float = 0.0
    normal_exposure: float = 1.0


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_realized_volatility(
    returns: pd.Series,
    window: int = 1
) -> pd.Series:
    """
    Compute realized volatility from log returns.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    window : int
        Window for realized volatility calculation
        
    Returns
    -------
    pd.Series
        Realized volatility series
    """
    # CRISIS: Use squared returns for realized volatility
    # Multiply by sqrt(252) for annualization (trading days)
    rv = (returns ** 2).rolling(window=window).mean() * 252
    return np.sqrt(rv)


def compute_har_features(
    returns: pd.Series,
    config: CrisisRegimeConfig
) -> pd.DataFrame:
    """
    Construct HAR (Heterogeneous Autoregressive) features.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    config : CrisisRegimeConfig
        Configuration object
        
    Returns
    -------
    pd.DataFrame
        DataFrame with HAR features
    """
    # CRISIS: Compute realized volatility at different time scales
    rv_daily = compute_realized_volatility(returns, config.volatility_window)
    rv_weekly = compute_realized_volatility(returns, config.weekly_window)
    rv_monthly = compute_realized_volatility(returns, config.monthly_window)
    
    # CRISIS: Create lagged features (strictly causal)
    features = pd.DataFrame({
        'rv_daily_lag1': rv_daily.shift(1),
        'rv_weekly_lag1': rv_weekly.shift(1),
        'rv_monthly_lag1': rv_monthly.shift(1),
        'rv_daily_lag5': rv_daily.shift(5),
        'rv_weekly_lag5': rv_weekly.shift(5),
        'rv_monthly_lag5': rv_monthly.shift(5),
        'rv_daily_lag22': rv_daily.shift(22),
        'rv_weekly_lag22': rv_weekly.shift(22),
        'rv_monthly_lag22': rv_monthly.shift(22)
    })
    
    return features


def compute_forward_volatility_targets(
    returns: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Compute forward realized volatility targets.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    horizon : int
        Forecast horizon in days
        
    Returns
    -------
    pd.Series
        Forward realized volatility targets
    """
    # CRISIS: Shift returns to create forward-looking targets
    # FIX: Correct forward volatility target alignment at time t
    # yt = sqrt(252 * mean(r[t+1:t+h]^2))
    # Target at index t should use future returns t+1..t+h
    forward_returns = returns.shift(-1)  # Start from t+1
    
    # Compute squared returns over the forward period
    forward_squared_returns = forward_returns ** 2
    
    # Mean over horizon days and annualize
    forward_rv = forward_squared_returns.rolling(window=horizon).mean() * 252
    forward_vol = np.sqrt(forward_rv)
    
    # FIX: Shift target forward to align with time t
    # Target at index t should be aligned with features at t
    forward_vol = forward_vol.shift(horizon-1)
    
    # FIX: Ensure stable column name
    forward_vol.name = f"fwd_vol_{horizon}"
    
    return forward_vol


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------

def build_crisis_dataset(
    raw_df: pd.DataFrame,
    config: CrisisRegimeConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature and target arrays for HAR-RV model.

    Parameters
    ----------
    raw_df : DataFrame
        Raw OHLCV data loaded from CSV.
    config : CrisisRegimeConfig
        Configuration object.

    Returns
    -------
    X_crisis : DataFrame
        Feature matrix for HAR-RV model.
    y_crisis : Series
        Forward volatility targets.
    forward_returns : Series
        Log returns for analysis.
    """
    # Ensure sorted by time
    df = raw_df.sort_values("date").reset_index(drop=True)
    
    # CRISIS: Compute log returns from close prices
    close = df["close"]
    log_returns = np.log(close / close.shift(1))
    # FIX: Remove silent bias - do not fillna(0), let NaNs exist
    # NaNs will be handled by dropping rows (training) or masking (inference)
    
    # Compute HAR features
    features = compute_har_features(log_returns, config)
    
    # Compute forward volatility targets
    targets = compute_forward_volatility_targets(log_returns, config.forecast_horizon)
    
    # Align features and targets, drop NaNs
    # CRISIS: Ensure no lookahead bias - only use past information
    valid_data = pd.concat([features, targets], axis=1).dropna()
    
    if len(valid_data) < config.min_train_samples:
        raise ValueError(
            f"Insufficient crisis samples ({len(valid_data)}) "
            f"for training; minimum required is {config.min_train_samples}."
        )
    
    X_crisis = valid_data.drop(columns=[targets.name]).copy()
    y_crisis = valid_data[targets.name].copy()
    forward_returns = log_returns.copy()
    
    return X_crisis, y_crisis, forward_returns


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def walk_forward_validation_crisis(
    X: pd.DataFrame,
    y: pd.Series,
    config: CrisisRegimeConfig
) -> List[Dict[str, Any]]:
    """
    Perform walk-forward validation on crisis regime samples.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Volatility targets
    config : CrisisRegimeConfig
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
        
        # Train Ridge regression model
        # FIX: Ridge constructor compatibility - remove random_state to prevent TypeError
        model = Ridge(alpha=config.ridge_alpha)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_val)
        
        # Compute regression metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        # Compute correlation (R-squared approximation)
        correlation = np.corrcoef(y_val, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        metrics = {
            "split": split,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "model": model,
            "y_val": y_val,
            "y_pred": y_pred
        }
        
        results.append(metrics)
    
    return results


def train_crisis_expert(
    X_crisis: pd.DataFrame,
    y_crisis: pd.Series,
    config: CrisisRegimeConfig,
) -> Tuple[Ridge, Dict[str, float], List[str]]:
    """
    Train Crisis regime HAR-RV model with walk-forward validation.

    Returns
    -------
    model : Ridge
        Fitted Ridge regression model.
    metrics : dict
        Walk-forward validation metrics.
    feature_columns : list
        List of feature column names.
    """
    # Walk-forward validation
    validation_results = walk_forward_validation_crisis(X_crisis, y_crisis, config)
    
    # FIX: Walk-forward validation must never silently return empty
    if len(validation_results) == 0:
        raise ValueError(
            f"Walk-forward validation produced 0 splits. "
            f"Reduce n_walk_forward_splits ({config.n_walk_forward_splits}) "
            f"or increase data size ({len(X_crisis)} samples)."
        )
    
    # Compute comprehensive metrics
    metrics = {
        "mean_rmse": np.mean([r["rmse"] for r in validation_results]),
        "std_rmse": np.std([r["rmse"] for r in validation_results]),
        "mean_mae": np.mean([r["mae"] for r in validation_results]),
        "std_mae": np.std([r["mae"] for r in validation_results]),
        "mean_correlation": np.mean([r["correlation"] for r in validation_results]),
        "std_correlation": np.std([r["correlation"] for r in validation_results]),
        "total_samples": sum([r["val_size"] for r in validation_results]),
        "n_splits": len(validation_results)
    }
    
    # Train final model on all crisis data
    # FIX: Ridge constructor compatibility - remove random_state to prevent TypeError
    final_model = Ridge(alpha=config.ridge_alpha)
    final_model.fit(X_crisis, y_crisis)
    
    return final_model, metrics, list(X_crisis.columns)


# ---------------------------------------------------------------------------
# Inference Functions
# ---------------------------------------------------------------------------

def predict_crisis_vol(
    df: pd.DataFrame,
    model: Ridge,
    config: CrisisRegimeConfig,
) -> pd.Series:
    """
    Generate volatility forecasts using trained HAR-RV model.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with OHLCV columns and date
    model : Ridge
        Trained Ridge regression model
    config : CrisisRegimeConfig
        Configuration object
        
    Returns
    -------
    pd.Series
        Volatility forecasts aligned with input data
    """
    # Ensure sorted by time
    df = df.sort_values("date").reset_index(drop=True)
    
    # CRISIS: Compute log returns from close prices
    close = df["close"]
    log_returns = np.log(close / close.shift(1))
    # FIX: Remove silent bias - do not fillna(0), let NaNs exist
    
    # Compute HAR features
    features = compute_har_features(log_returns, config)
    
    # FIX: Ensure feature columns match training with missing column guard
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        
        # Check for missing columns
        missing_cols = [col for col in expected_features if col not in features.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns: {missing_cols}. "
                f"Expected: {expected_features}, Available: {list(features.columns)}"
            )
        
        # Reorder features to match training order
        features = features[expected_features]
    
    # FIX: Predict only on rows where features are fully non-NaN
    valid_rows = features.notna().all(axis=1)
    
    # Generate volatility forecasts
    vol_forecasts = pd.Series(np.nan, index=df.index)
    if valid_rows.any():
        vol_forecasts[valid_rows] = model.predict(features[valid_rows])
    
    return vol_forecasts


def crisis_risk_off_exposure(
    vol_forecasts: pd.Series,
    config: CrisisRegimeConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply deterministic risk-off rule based on volatility forecasts.

    Parameters
    ----------
    vol_forecasts : pd.Series
        Volatility forecasts
    config : CrisisRegimeConfig
        Configuration object
        
    Returns
    -------
    crisis_flag : pd.Series
        Boolean flag indicating crisis regime
    exposure : pd.Series
        Recommended exposure level
    """
    # CRISIS: Compute rolling threshold using high quantile
    # FIX: Remove lookahead in risk-off threshold - use only past forecasts
    # Compute rolling quantile on vol_forecasts.shift(1), not vol_forecasts
    past_forecasts = vol_forecasts.shift(1)
    rolling_threshold = past_forecasts.rolling(
        window=config.crisis_lookback,
        min_periods=50
    ).quantile(config.crisis_quantile)
    
    # CRISIS: Crisis flag when forecast exceeds threshold
    crisis_flag = vol_forecasts > rolling_threshold
    
    # CRISIS: Apply exposure rules
    exposure = np.where(
        crisis_flag,
        config.min_exposure,  # Crisis: minimum exposure
        config.normal_exposure  # Normal: full exposure
    )
    
    exposure_series = pd.Series(exposure, index=vol_forecasts.index)
    
    return crisis_flag, exposure_series


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_crisis_models(
    model: Ridge,
    metrics: Dict[str, float],
    feature_names: Tuple[str, ...],
    config: CrisisRegimeConfig,
    output_dir: str = "Model",
) -> None:
    """
    Save trained Crisis Expert model and metadata to disk.

    Files written
    ------------
    - crisis_har_model.pkl : Ridge regression model
    - crisis_metadata.json : feature names, metrics, config
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "crisis_har_model.pkl")
    metadata_path = os.path.join(output_dir, "crisis_metadata.json")

    joblib.dump(model, model_path)

    # CRISIS: Save configuration and metadata
    metadata = {
        "feature_names": list(feature_names),
        "metrics": metrics,
        "model_type": "CrisisExpert",
        "method": "HAR-RV",
        "config": {
            "volatility_window": config.volatility_window,
            "weekly_window": config.weekly_window,
            "monthly_window": config.monthly_window,
            "forecast_horizon": config.forecast_horizon,
            "ridge_alpha": config.ridge_alpha,
            "crisis_quantile": config.crisis_quantile,
            "crisis_lookback": config.crisis_lookback,
            "min_exposure": config.min_exposure,
            "normal_exposure": config.normal_exposure
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_crisis_models(model_dir: str = "Model") -> Dict[str, Any]:
    """
    Load trained Crisis Expert model and metadata.
    
    Parameters
    ----------
    model_dir : str
        Directory containing trained models
        
    Returns
    -------
    dict
        Dictionary with model, metadata, and config
    """
    model_path = os.path.join(model_dir, "crisis_har_model.pkl")
    metadata_path = os.path.join(model_dir, "crisis_metadata.json")
    
    if not all(os.path.exists(p) for p in [model_path, metadata_path]):
        raise FileNotFoundError(f"Crisis Expert model files not found in {model_dir}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # CRISIS: Reconstruct configuration from metadata
    config_data = metadata.get("config", {})
    crisis_config = CrisisRegimeConfig(
        volatility_window=config_data.get("volatility_window", 1),
        weekly_window=config_data.get("weekly_window", 5),
        monthly_window=config_data.get("monthly_window", 22),
        forecast_horizon=config_data.get("forecast_horizon", HORIZON_DAYS),
        ridge_alpha=config_data.get("ridge_alpha", 1.0),
        crisis_quantile=config_data.get("crisis_quantile", 0.90),
        crisis_lookback=config_data.get("crisis_lookback", 252),
        min_exposure=config_data.get("min_exposure", 0.0),
        normal_exposure=config_data.get("normal_exposure", 1.0)
    )
    
    return {
        "model": model,
        "metadata": metadata,
        "feature_columns": metadata["feature_names"],
        "config": crisis_config
    }


# ---------------------------------------------------------------------------
# Complete Inference Pipeline
# ---------------------------------------------------------------------------

def get_crisis_signals(
    df: pd.DataFrame,
    artifacts: Dict[str, Any] | None = None,
    model_dir: str = "Model"
) -> pd.DataFrame:
    """
    Generate complete crisis signals with volatility forecasts and exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with OHLCV columns
    artifacts : dict, optional
        Pre-loaded model artifacts
    model_dir : str
        Directory containing trained models
        
    Returns
    -------
    pd.DataFrame
        DataFrame with forecasts, crisis flags, and exposure
    """
    # Load models if not provided
    if artifacts is None:
        artifacts = load_crisis_models(model_dir)
    
    model = artifacts["model"]
    config = artifacts["config"]
    
    # FIX: Ensure train/inference output alignment in get_crisis_signals
    # Use same sorted/reset dataframe as predictions for output construction
    df_sorted = df.sort_values("date").reset_index(drop=True)
    
    # Generate volatility forecasts (uses sorted df internally)
    vol_forecasts = predict_crisis_vol(df_sorted, model, config)
    
    # Apply risk-off rule
    crisis_flag, exposure = crisis_risk_off_exposure(vol_forecasts, config)
    
    # Create result DataFrame using sorted data to maintain alignment
    result = pd.DataFrame({
        "date": df_sorted["date"],
        "close": df_sorted["close"],
        "volatility_forecast": vol_forecasts,
        "crisis_flag": crisis_flag,
        "exposure": exposure
    })
    
    return result

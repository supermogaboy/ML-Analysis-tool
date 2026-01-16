"""
Crisis Regime Specialist Training
===============================

This module implements the training pipeline for the Crisis Regime Specialist.
It handles data loading, HAR-RV feature engineering, model training with
walk-forward validation, and model persistence.

The actual model implementation is in Model/Crisis_Expert.py
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
# Top-level training script
# ---------------------------------------------------------------------------

def load_qqq_processed_csv(path: str) -> pd.DataFrame:
    """
    Load processed QQQ CSV from existing data pipeline.

    Expected columns (from processed pipeline):
    - date, close, high, low, volume, and processed features
    """
    df = pd.read_csv(path)
    # Normalize column names to lower snake_case if needed
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    
    # Ensure date is datetime
    if df["date"].dtype == "object":
        df["date"] = pd.to_datetime(df["date"])
    
    return df


def train_crisis_regime_specialist(
    csv_path: str = "data/processed/QQQ_1d_processed.csv",
    output_dir: str = "Model",
    config: CrisisRegimeConfig | None = None,
) -> Dict[str, float]:
    """
    End-to-end training pipeline for the Crisis Regime specialist.

    Steps
    -----
    1. Load processed QQQ CSV (with OHLCV data).
    2. Compute HAR-RV features from log returns.
    3. Compute forward volatility targets.
    4. Train Ridge regression with walk-forward validation.
    5. Evaluate regression metrics (RMSE, MAE, correlation).
    6. Refit on all data and persist models + metadata to disk.

    Returns
    -------
    dict
        Metrics from the walk-forward validation.
    """
    if config is None:
        config = CrisisRegimeConfig()

    raw_df = load_qqq_processed_csv(csv_path)

    # Import model functions from Model directory
    sys.path.append(os.path.join(PROJECT_ROOT, "Model"))
    from Crisis_Expert import (
        build_crisis_dataset, 
        train_crisis_expert, 
        save_crisis_models
    )

    X_crisis, y_crisis, forward_returns = build_crisis_dataset(
        raw_df=raw_df,
        config=config,
    )

    model, metrics, feature_columns = train_crisis_expert(
        X_crisis=X_crisis,
        y_crisis=y_crisis,
        config=config,
    )

    save_crisis_models(
        model=model,
        metrics=metrics,
        feature_names=tuple(feature_columns),
        config=config,
        output_dir=output_dir,
    )

    return metrics


if __name__ == "__main__":
    # Simple CLI entry point for training the Crisis specialist.
    # Ensure we're in the project root for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Default path to processed data
    processed_csv = os.path.join(project_root, "data", "processed", "QQQ_1d_processed.csv")
    
    if not os.path.exists(processed_csv):
        print(f"Error: Processed data not found at {processed_csv}")
        print("Please run the data pipeline first:")
        print("  python src/run_data_pipeline.py")
        sys.exit(1)
    
    print(f"Loading processed data from: {processed_csv}")
    metrics_out = train_crisis_regime_specialist(csv_path=processed_csv)
    print("\n" + "="*60)
    print("Crisis Regime specialist trained successfully!")
    print("="*60)
    print("Walk-forward validation metrics:")
    for k, v in metrics_out.items():
        print(f"  {k}: {v}")
    print("="*60)

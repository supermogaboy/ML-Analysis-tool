"""
Chop Regime Specialist Training
==============================

This module implements the training pipeline for the Chop Regime Specialist.
It handles data loading, feature engineering, model training with walk-forward
validation, and model persistence.

The actual model implementation is in Model/Chop_Expert.py
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
# Top-level training script
# ---------------------------------------------------------------------------

def load_qqq_processed_csv(path: str) -> pd.DataFrame:
    """
    Load processed QQQ CSV from existing data pipeline.

    Expected columns (from processed pipeline):
    - date, close, high, low, volume, and processed features
    - Optional: fwd_ret, y_class, U, D, regime
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


def train_chop_regime_specialist(
    csv_path: str = "data/processed/QQQ_1d_processed.csv",
    output_dir: str = "Model",
    config: ChopRegimeConfig | None = None,
) -> Dict[str, float]:
    """
    End-to-end training pipeline for the Chop Regime specialist.

    Steps
    -----
    1. Load processed QQQ CSV (with features and labels).
    2. Extract/map features from processed data.
    3. Extract/compute labels from processed data.
    4. Filter to chop regime observations only using z-score filtering.
    5. Train XGBoost classifier with walk-forward validation.
    6. Calibrate probabilities using validation predictions.
    7. Refit on all chop data and persist models + metadata to disk.

    Returns
    -------
    dict
        Metrics from the walk-forward validation.
    """
    if config is None:
        config = ChopRegimeConfig()

    raw_df = load_qqq_processed_csv(csv_path)

    # Import model functions from Model directory
    sys.path.append(os.path.join(PROJECT_ROOT, "Model"))
    from Chop_Expert import (
        build_chop_dataset, 
        train_chop_specialist, 
        save_chop_models
    )

    X_chop, y_chop, chop_mask, forward_returns = build_chop_dataset(
        raw_df=raw_df,
        config=config,
    )

    model, calibrator, feature_columns, metrics = train_chop_specialist(
        X_chop=X_chop,
        y_chop=y_chop,
        config=config,
    )

    save_chop_models(
        model=model,
        calibrator=calibrator,
        feature_names=tuple(feature_columns),
        metrics=metrics,
        config=config,
        output_dir=output_dir,
    )

    return metrics


if __name__ == "__main__":
    # Simple CLI entry point for training the Chop specialist.
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
    metrics_out = train_chop_regime_specialist(csv_path=processed_csv)
    print("\n" + "="*60)
    print("Chop Regime specialist trained successfully!")
    print("="*60)
    print("Walk-forward validation metrics:")
    for k, v in metrics_out.items():
        print(f"  {k}: {v}")
    print("="*60)

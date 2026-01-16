#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import joblib

# Configuration
DATA_PATH = "data/processed/QQQ_1d_processed.csv"
MODELS_DIR = "models"
HORIZON_DAYS = 5
REGIMES = ["trend_risk", "volatile_event", "calm", "crisis"]
RANDOM_SEED = 42

def load_and_prepare_data():
    """Load data and create up/down targets"""
    df = pd.read_csv(DATA_PATH)
    
    # Ensure regime column is lowercase
    df['regime'] = df['regime'].str.lower()
    
    # Create binary targets for up/down
    df['target_up'] = (df['fwd_ret'] > 0).astype(int)
    df['target_down'] = (df['fwd_ret'] < 0).astype(int)
    
    # Create magnitude targets (conditional on direction)
    df['up_magnitude'] = np.where(df['fwd_ret'] > 0, df['fwd_ret'], 0)
    df['down_magnitude'] = np.where(df['fwd_ret'] < 0, df['fwd_ret'], 0)
    
    return df

def get_feature_columns(df):
    """Define feature columns (exclude target and metadata)"""
    exclude_cols = [
        'date', 'regime', 'fwd_ret', 'y_class', 'U', 'D',
        'target_up', 'target_down', 'up_magnitude', 'down_magnitude'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def train_regime_models(df, regime, feature_cols):
    """Train up/down models for a specific regime"""
    print(f"\nTraining models for {regime.upper()} regime...")
    
    # Filter data for this regime
    regime_data = df[df['regime'] == regime].copy()
    
    if len(regime_data) < 100:
        print(f"Warning: Only {len(regime_data)} samples for {regime} regime")
        return None, None, {}
    
    # Prepare features
    X = regime_data[feature_cols].fillna(0)
    
    # Time series split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    # Up models
    y_up_binary = regime_data['target_up'].iloc[:split_idx]
    y_up_binary_test = regime_data['target_up'].iloc[split_idx:]
    y_up_mag = regime_data['up_magnitude'].iloc[:split_idx]
    y_up_mag_test = regime_data['up_magnitude'].iloc[split_idx:]
    
    # Down models  
    y_down_binary = regime_data['target_down'].iloc[:split_idx]
    y_down_binary_test = regime_data['target_down'].iloc[split_idx:]
    y_down_mag = regime_data['down_magnitude'].iloc[:split_idx]
    y_down_mag_test = regime_data['down_magnitude'].iloc[split_idx:]
    
    # Train up confidence model
    up_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_SEED
    )
    up_clf.fit(X_train, y_up_binary)
    
    # Train up magnitude model (only on positive returns)
    up_mag_mask = y_up_mag > 0
    if up_mag_mask.sum() > 10:
        up_reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=RANDOM_SEED
        )
        up_reg.fit(X_train[up_mag_mask], y_up_mag[up_mag_mask])
    else:
        up_reg = None
    
    # Train down confidence model
    down_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_SEED
    )
    down_clf.fit(X_train, y_down_binary)
    
    # Train down magnitude model (only on negative returns)
    down_mag_mask = y_down_mag < 0
    if down_mag_mask.sum() > 10:
        down_reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=RANDOM_SEED
        )
        down_reg.fit(X_train[down_mag_mask], y_down_mag[down_mag_mask])
    else:
        down_reg = None
    
    # Calculate metrics
    metrics = {}
    
    # Up metrics
    if len(np.unique(y_up_binary_test)) > 1:
        up_pred_proba = up_clf.predict_proba(X_test)[:, 1]
        metrics['up_auc'] = roc_auc_score(y_up_binary_test, up_pred_proba)
    else:
        metrics['up_auc'] = 0.5
    
    if up_reg is not None and (y_up_mag_test > 0).sum() > 0:
        up_test_mask = y_up_mag_test > 0
        up_pred_mag = up_reg.predict(X_test[up_test_mask])
        metrics['up_mae'] = mean_absolute_error(y_up_mag_test[up_test_mask], up_pred_mag)
        metrics['up_rmse'] = np.sqrt(mean_squared_error(y_up_mag_test[up_test_mask], up_pred_mag))
    else:
        metrics['up_mae'] = 0.0
        metrics['up_rmse'] = 0.0
    
    # Down metrics
    if len(np.unique(y_down_binary_test)) > 1:
        down_pred_proba = down_clf.predict_proba(X_test)[:, 1]
        metrics['down_auc'] = roc_auc_score(y_down_binary_test, down_pred_proba)
    else:
        metrics['down_auc'] = 0.5
    
    if down_reg is not None and (y_down_mag_test < 0).sum() > 0:
        down_test_mask = y_down_mag_test < 0
        down_pred_mag = down_reg.predict(X_test[down_test_mask])
        metrics['down_mae'] = mean_absolute_error(y_down_mag_test[down_test_mask], down_pred_mag)
        metrics['down_rmse'] = np.sqrt(mean_squared_error(y_down_mag_test[down_test_mask], down_pred_mag))
    else:
        metrics['down_mae'] = 0.0
        metrics['down_rmse'] = 0.0
    
    metrics['n_samples'] = len(regime_data)
    metrics['n_train'] = len(X_train)
    metrics['n_test'] = len(X_test)
    
    return (up_clf, up_reg), (down_clf, down_reg), metrics

def save_models(regime, up_models, down_models, feature_cols, metrics):
    """Save models and metadata for a regime"""
    regime_dir = Path(MODELS_DIR) / regime
    regime_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    joblib.dump(up_models[0], regime_dir / "model_up_clf.pkl")  # confidence
    if up_models[1] is not None:
        joblib.dump(up_models[1], regime_dir / "model_up_reg.pkl")  # magnitude
    else:
        # Create dummy model for consistency
        dummy = GradientBoostingRegressor(n_estimators=1, max_depth=1)
        dummy.fit([[0]], [0])
        joblib.dump(dummy, regime_dir / "model_up_reg.pkl")
    
    joblib.dump(down_models[0], regime_dir / "model_down_clf.pkl")  # confidence
    if down_models[1] is not None:
        joblib.dump(down_models[1], regime_dir / "model_down_reg.pkl")  # magnitude
    else:
        # Create dummy model for consistency
        dummy = GradientBoostingRegressor(n_estimators=1, max_depth=1)
        dummy.fit([[0]], [0])
        joblib.dump(dummy, regime_dir / "model_down_reg.pkl")
    
    # Save feature list
    with open(regime_dir / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    # Save config
    config = {
        "regime": regime,
        "horizon_days": HORIZON_DAYS,
        "training_date": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED
    }
    with open(regime_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    with open(regime_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved models to {regime_dir}")

def main():
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print(f"Total samples: {len(df)}")
    print(f"Regime distribution:")
    print(df['regime'].value_counts())
    
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")
    
    # Train models for each regime
    for regime in REGIMES:
        if regime not in df['regime'].values:
            print(f"Warning: No data found for {regime} regime")
            continue
            
        up_models, down_models, metrics = train_regime_models(df, regime, feature_cols)
        
        if up_models is not None:
            save_models(regime, up_models, down_models, feature_cols, metrics)
            print(f"{regime.upper()} metrics: {metrics}")
        else:
            print(f"Skipping {regime} due to insufficient data")
    
    print(f"\nTraining complete! Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, Any

MODELS_DIR = "models"

def load_regime_models(regime_name: str):
    """Load models for a specific regime"""
    regime_dir = Path(MODELS_DIR) / regime_name.lower()
    
    if not regime_dir.exists():
        raise FileNotFoundError(f"No models found for regime: {regime_name}")
    
    # Load feature list
    with open(regime_dir / "feature_list.json", "r") as f:
        feature_cols = json.load(f)
    
    # Load models
    up_clf = joblib.load(regime_dir / "model_up_clf.pkl")
    up_reg = joblib.load(regime_dir / "model_up_reg.pkl")
    down_clf = joblib.load(regime_dir / "model_down_clf.pkl")
    down_reg = joblib.load(regime_dir / "model_down_reg.pkl")
    
    return {
        'feature_cols': feature_cols,
        'up_clf': up_clf,
        'up_reg': up_reg,
        'down_clf': down_clf,
        'down_reg': down_reg
    }

def predict_regime(regime_name: str, X_row: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate up/down forecasts for a specific regime.
    
    Args:
        regime_name: One of "trend", "chop", "calm", "crisis"
        X_row: DataFrame with a single row of features
        
    Returns:
        Dict with up/down forecasts containing expected_return and confidence
    """
    # Load models
    models = load_regime_models(regime_name)
    
    # Ensure X_row has the right features in the right order
    X = X_row[models['feature_cols']].fillna(0)
    
    # Up predictions
    up_confidence = models['up_clf'].predict_proba(X)[0, 1]  # P(up)
    up_magnitude = models['up_reg'].predict(X)[0]  # Expected positive return
    up_expected_return = up_confidence * max(0, up_magnitude)  # E[r | up] * P(up)
    
    # Down predictions  
    down_confidence = models['down_clf'].predict_proba(X)[0, 1]  # P(down)
    down_magnitude = models['down_reg'].predict(X)[0]  # Expected negative return
    down_expected_return = down_confidence * min(0, down_magnitude)  # E[r | down] * P(down)
    
    return {
        "regime": regime_name.upper(),
        "up": {
            "expected_return": float(up_expected_return),
            "confidence": float(up_confidence)
        },
        "down": {
            "expected_return": float(down_expected_return), 
            "confidence": float(down_confidence)
        }
    }

def get_regime_info(regime_name: str) -> Dict[str, Any]:
    """Get metadata about a regime's models"""
    regime_dir = Path(MODELS_DIR) / regime_name.lower()
    
    if not regime_dir.exists():
        raise FileNotFoundError(f"No models found for regime: {regime_name}")
    
    # Load config and metrics
    with open(regime_dir / "config.json", "r") as f:
        config = json.load(f)
    
    with open(regime_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    
    with open(regime_dir / "feature_list.json", "r") as f:
        feature_cols = json.load(f)
    
    return {
        "config": config,
        "metrics": metrics,
        "n_features": len(feature_cols),
        "features": feature_cols
    }

def list_available_regimes():
    """List all available trained regimes"""
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        return []
    
    regimes = []
    for item in models_path.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            regimes.append(item.name.upper())
    
    return sorted(regimes)

# Example usage
if __name__ == "__main__":
    # Test inference if models exist
    regimes = list_available_regimes()
    if regimes:
        print(f"Available regimes: {regimes}")
        
        # Create dummy data for testing
        dummy_features = {
            'adj_close': [100.0],
            'close': [100.0], 
            'high': [105.0],
            'low': [95.0],
            'open': [98.0],
            'volume': [1000000],
            'logret_1': [0.01],
            'logret_5': [0.02],
            'logret_10': [0.03],
            'vol_10': [0.015],
            'vol_20': [0.018],
            'ema_20': [99.0],
            'ema_50': [98.0],
            'ema_200': [97.0],
            'trend_20_50': [0.01],
            'trend_px_200': [0.03],
            'rsi_14': [55.0],
            'bb_z_20': [0.5],
            'atr_14': [2.0],
            'atrp_14': [0.02],
            'vol_sma20': [0.017],
            'vol_std20': [0.005],
            'vol_z20': [0.3],
            'dd_63': [0.05],
            'dist_252_high': [0.1]
        }
        
        X_test = pd.DataFrame(dummy_features)
        
        for regime in regimes:
            try:
                result = predict_regime(regime, X_test)
                print(f"\n{regime} prediction:")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error predicting {regime}: {e}")
    else:
        print("No trained models found. Run train_regime_models.py first.")

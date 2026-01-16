import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add classifier to path
sys.path.append(str(Path(__file__).parent.parent / "classifier"))
from RegimeClassifier import RegimeClassifier

def detect_regimes(df, use_classifier=True, model_path=None):
    """
    Detect market regimes using either trained classifier or rule-based approach
    
    Args:
        df: DataFrame with price and volume data
        use_classifier: Whether to use RegimeClassifier (recommended)
        model_path: Path to saved RegimeClassifier model
    
    Returns:
        DataFrame with regime column added
    """
    if use_classifier:
        # Use the trained RegimeClassifier
        classifier = RegimeClassifier()
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            classifier.load(model_path)
            print(f"Loaded pre-trained regime classifier from {model_path}")
        else:
            # Train on the data
            print("Training regime classifier on data...")
            classifier.fit(df)
            
            # Save the trained model
            if model_path:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                classifier.save(model_path)
                print(f"Saved trained regime classifier to {model_path}")
        
        # Predict regimes
        regimes = classifier.predict(df)
        
        # Map to lowercase names to match existing pipeline
        regime_mapping = {
            "Calm": "calm",
            "General Trend with Risk": "trend_risk", 
            "Volatile Event": "volatile_event",
            "Crisis": "crisis"
        }
        
        df['regime'] = regimes.map(regime_mapping)
        
        # Fill any NaN values with 'calm' as default
        df['regime'] = df['regime'].fillna('calm')
        
    else:
        # Fallback to rule-based approach
        df = _detect_regimes_rule_based(df)
    
    return df

def _detect_regimes_rule_based(df, vol_window=20, trend_window=50, crisis_threshold=0.03, calm_threshold=0.01):
    """
    Rule-based regime detection (original approach as fallback)
    """
    # Calculate rolling volatility if not present
    if 'vol_20' not in df.columns:
        df['vol_20'] = df['logret_1'].rolling(vol_window).std()
    
    # Calculate trend strength if not present
    if 'trend_20_50' not in df.columns:
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=trend_window).mean()
        df['trend_20_50'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    
    # Initialize regime column
    df['regime'] = 'calm'  # Default regime
    
    # Crisis regime: High volatility
    crisis_mask = df['vol_20'] > crisis_threshold
    df.loc[crisis_mask, 'regime'] = 'crisis'
    
    # Trend regime: Strong trend with moderate volatility
    trend_mask = (
        (df['vol_20'] <= crisis_threshold) & 
        (df['vol_20'] > calm_threshold) & 
        (np.abs(df['trend_20_50']) > 0.02)
    )
    df.loc[trend_mask, 'regime'] = 'trend_risk'
    
    # Volatile event regime: Very high volatility spikes
    vol_spike_mask = df['vol_20'] > (df['vol_20'].rolling(252).mean() + 2 * df['vol_20'].rolling(252).std())
    df.loc[vol_spike_mask, 'regime'] = 'volatile_event'
    
    # Chop regime: Low volatility, no clear trend
    chop_mask = (
        (df['vol_20'] <= calm_threshold) & 
        (np.abs(df['trend_20_50']) <= 0.01)
    )
    df.loc[chop_mask, 'regime'] = 'chop'
    
    return df

def add_regime_features(df):
    """Add additional regime-specific features"""
    # Regime encoding for ML models
    regime_dummies = pd.get_dummies(df['regime'], prefix='regime')
    df = pd.concat([df, regime_dummies], axis=1)
    
    # Regime-specific volatility adjustments
    df['vol_regime_adj'] = df['vol_20'] / df.groupby('regime')['vol_20'].transform('mean')
    
    return df

def smooth_regimes(df, smoothing_window=3):
    """Apply smoothing to reduce regime switching noise"""
    # Forward fill dominant regime
    df['regime_smooth'] = df['regime'].copy()
    
    # Apply majority filter over smoothing window
    for i in range(smoothing_window, len(df) - smoothing_window):
        window_regimes = df['regime'].iloc[i-smoothing_window:i+smoothing_window+1]
        most_common = window_regimes.mode().iloc[0] if len(window_regimes.mode()) > 0 else df['regime'].iloc[i]
        df.loc[df.index[i], 'regime_smooth'] = most_common
    
    return df

def train_regime_classifier(df, model_path="models/regime_classifier.pkl"):
    """
    Train and save a RegimeClassifier model
    
    Args:
        df: DataFrame with price data
        model_path: Path to save the trained model
    
    Returns:
        Trained RegimeClassifier instance
    """
    # Prepare data for classifier (needs Adj Close column)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['close']  # Use close if Adj Close not available
    
    classifier = RegimeClassifier()
    classifier.fit(df)
    
    # Save the model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_path)
    print(f"Regime classifier saved to {model_path}")
    
    return classifier

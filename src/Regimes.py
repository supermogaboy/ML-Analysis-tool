import pandas as pd
import numpy as np

def detect_regimes(df, vol_window=20, trend_window=50, crisis_threshold=0.03, calm_threshold=0.01):
    """
    Detect market regimes based on volatility and trend characteristics
    
    Args:
        df: DataFrame with price and volatility data
        vol_window: Window for volatility calculation
        trend_window: Window for trend calculation
        crisis_threshold: Volatility threshold for crisis regime
        calm_threshold: Volatility threshold for calm regime
    
    Returns:
        DataFrame with regime column added
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

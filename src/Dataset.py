import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from .Features import create_features, create_labels
from .Regimes import detect_regimes

def load_raw_data(symbol="QQQ", start_date="2000-01-01", end_date="2024-01-01"):
    """Load raw OHLCV data using yfinance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data = data.reset_index()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_data(df, horizon_days=5, flat_band=0.001):
    """
    Process raw OHLCV data into features and labels
    
    Args:
        df: Raw OHLCV DataFrame
        horizon_days: Forward return horizon for labels
        flat_band: Threshold for flat classification
    
    Returns:
        Processed DataFrame with features and labels
    """
    # Create technical features
    df = create_features(df, price_col='close')
    
    # Create classification labels
    df = create_labels(df, flat_band)
    
    # Detect market regimes
    df = detect_regimes(df)
    
    # Remove rows with NaN values (from rolling calculations)
    df = df.dropna()
    
    # Ensure regime is lowercase
    df = df.copy()
    df['regime'] = df['regime'].str.lower()
    
    return df

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def load_processed_data(input_path):
    """Load processed data from CSV"""
    return pd.read_csv(input_path)

def create_train_test_split(df, test_ratio=0.2):
    """
    Create time-series aware train/test split
    
    Args:
        df: Processed DataFrame
        test_ratio: Fraction of data for testing
    
    Returns:
        train_df, test_df
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

def get_feature_columns(df, exclude_cols=None):
    """
    Get list of feature columns for ML models
    
    Args:
        df: DataFrame
        exclude_cols: List of columns to exclude
    
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = [
            'date', 'regime', 'fwd_ret', 'y_class', 'U', 'D',
            'target_up', 'target_down', 'up_magnitude', 'down_magnitude'
        ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def validate_data(df):
    """Validate processed data for common issues"""
    issues = []
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"Missing values in columns: {missing_cols}")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_df).any()
    inf_cols = numeric_df.columns[inf_mask].tolist()
    if inf_cols:
        issues.append(f"Infinite values in columns: {inf_cols}")
    
    # Check regime distribution
    regime_counts = df['regime'].value_counts()
    if len(regime_counts) < 2:
        issues.append("Only one regime found - check regime detection")
    
    # Check forward returns
    if 'fwd_ret' not in df.columns:
        issues.append("Forward returns (fwd_ret) not found")
    else:
        if df['fwd_ret'].isnull().all():
            issues.append("All forward returns are null")
    
    return issues

def summarize_dataset(df):
    """Print summary statistics of the dataset"""
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Features: {len([col for col in df.columns if col not in ['date', 'regime', 'fwd_ret', 'y_class', 'U', 'D']])}")
    print()
    print("Regime Distribution:")
    print(df['regime'].value_counts())
    print()
    print("Label Distribution:")
    if 'y_class' in df.columns:
        print(df['y_class'].value_counts().sort_index())
    print()
    print("Forward Return Stats:")
    if 'fwd_ret' in df.columns:
        print(df['fwd_ret'].describe())
    print("=" * 50)

import pandas as pd
import numpy as np

def calculate_returns(df, price_col='close'):
    """Calculate log returns for various horizons"""
    df[f'logret_1'] = np.log(df[price_col] / df[price_col].shift(1))
    df[f'logret_5'] = np.log(df[price_col] / df[price_col].shift(5))
    df[f'logret_10'] = np.log(df[price_col] / df[price_col].shift(10))
    return df

def calculate_volatility(df, price_col='close'):
    """Calculate rolling volatility measures"""
    # Rolling standard deviation of returns
    df['vol_10'] = df['logret_1'].rolling(10).std()
    df['vol_20'] = df['logret_1'].rolling(20).std()
    
    # Volatility SMA and standard deviation
    df['vol_sma20'] = df['vol_20'].rolling(20).mean()
    df['vol_std20'] = df['vol_20'].rolling(20).std()
    df['vol_z20'] = (df['vol_20'] - df['vol_sma20']) / df['vol_std20']
    
    return df

def calculate_moving_averages(df, price_col='close'):
    """Calculate exponential moving averages"""
    df['ema_20'] = df[price_col].ewm(span=20).mean()
    df['ema_50'] = df[price_col].ewm(span=50).mean()
    df['ema_200'] = df[price_col].ewm(span=200).mean()
    
    # Trend indicators
    df['trend_20_50'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    df['trend_px_200'] = (df[price_col] - df['ema_200']) / df['ema_200']
    
    return df

def calculate_rsi(df, price_col='close', periods=14):
    """Calculate RSI"""
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, price_col='close', periods=20, std_dev=2):
    """Calculate Bollinger Bands and Z-score"""
    sma = df[price_col].rolling(periods).mean()
    std = df[price_col].rolling(periods).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    df['bb_z_20'] = (df[price_col] - sma) / std
    return df

def calculate_atr(df, periods=14):
    """Calculate Average True Range and ATR percentage"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(periods).mean()
    df['atrp_14'] = df['atr_14'] / df['close']
    
    return df

def calculate_drawdown(df, price_col='close', window=63):
    """Calculate drawdown from recent highs"""
    rolling_max = df[price_col].rolling(window=window).max()
    df['dd_63'] = (df[price_col] - rolling_max) / rolling_max
    return df

def calculate_distance_from_high(df, price_col='close', window=252):
    """Calculate distance from 52-week high"""
    rolling_max = df[price_col].rolling(window=window).max()
    df['dist_252_high'] = (df[price_col] - rolling_max) / rolling_max
    return df

def calculate_forward_returns(df, price_col='close', horizon=5):
    """Calculate forward returns for target variable"""
    df['fwd_ret'] = np.log(df[price_col].shift(-horizon) / df[price_col])
    return df

def create_features(df, price_col='close'):
    """Create all technical features"""
    df = calculate_returns(df, price_col)
    df = calculate_volatility(df, price_col)
    df = calculate_moving_averages(df, price_col)
    df = calculate_rsi(df, price_col)
    df = calculate_bollinger_bands(df, price_col)
    df = calculate_atr(df)
    df = calculate_drawdown(df, price_col)
    df = calculate_distance_from_high(df, price_col)
    df = calculate_forward_returns(df, price_col)
    
    return df

def create_labels(df, flat_band=0.001):
    """Create classification labels"""
    df['y_class'] = 0  # Default to neutral/flat
    
    # Up moves
    df.loc[df['fwd_ret'] > flat_band, 'y_class'] = 2
    df['U'] = (df['fwd_ret'] > flat_band).astype(float)
    
    # Down moves  
    df.loc[df['fwd_ret'] < -flat_band, 'y_class'] = 1
    df['D'] = (df['fwd_ret'] < -flat_band).astype(float)
    
    return df

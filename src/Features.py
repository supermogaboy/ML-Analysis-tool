# src/features.py
import numpy as np
import pandas as pd

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def bollinger_z(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - sma) / (2 * (std + 1e-12))

def max_drawdown(close: pd.Series, window: int = 63) -> pd.Series:
    roll_max = close.rolling(window).max()
    dd = (close / (roll_max + 1e-12)) - 1.0
    return dd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    df["logret_1"] = np.log(c).diff()
    df["logret_5"] = np.log(c).diff(5)
    df["logret_10"] = np.log(c).diff(10)

    df["vol_10"] = df["logret_1"].rolling(10).std()
    df["vol_20"] = df["logret_1"].rolling(20).std()

    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    df["trend_20_50"] = (df["ema_20"] - df["ema_50"]) / (c + 1e-12)
    df["trend_px_200"] = (c - df["ema_200"]) / (c + 1e-12)

    df["rsi_14"] = rsi(c, 14)
    df["bb_z_20"] = bollinger_z(c, 20)

    df["atr_14"] = atr(df, 14)
    df["atrp_14"] = df["atr_14"] / (c + 1e-12)

    vol = df["volume"].replace(0, np.nan)
    df["vol_sma20"] = vol.rolling(20).mean()
    df["vol_std20"] = vol.rolling(20).std()
    df["vol_z20"] = (vol - df["vol_sma20"]) / (df["vol_std20"] + 1e-12)

    df["dd_63"] = max_drawdown(c, 63)
    df["dist_252_high"] = (c / (c.rolling(252).max() + 1e-12)) - 1.0

    return df

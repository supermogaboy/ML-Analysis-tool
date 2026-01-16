import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# DOWNLOAD RAW DATA
# ==========================================

df = yf.download("QQQ", start="2000-01-01", end="2025-12-31")

# Flatten MultiIndex if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print("RAW COLUMNS:", df.columns.tolist())

# Normalize column names
df.columns = df.columns.str.lower()

# Select available columns (NO adj close)
df = df[["open", "high", "low", "close", "volume"]].dropna()

# ==========================================
# RETURNS
# ==========================================

df["logret_1"]  = np.log(df["close"] / df["close"].shift(1))
df["logret_5"]  = np.log(df["close"] / df["close"].shift(5))
df["logret_10"] = np.log(df["close"] / df["close"].shift(10))

# ==========================================
# VOLATILITY
# ==========================================

df["vol_10"] = df["logret_1"].rolling(10).std()
df["vol_20"] = df["logret_1"].rolling(20).std()
df["vol_z20"] = (
    df["vol_20"] - df["vol_20"].rolling(252).mean()
) / df["vol_20"].rolling(252).std()

# ==========================================
# TREND
# ==========================================

df["ema_20"]  = df["close"].ewm(span=20).mean()
df["ema_50"]  = df["close"].ewm(span=50).mean()
df["ema_200"] = df["close"].ewm(span=200).mean()

df["trend_20_50"]  = df["ema_20"] / df["ema_50"] - 1
df["trend_px_200"] = df["close"] / df["ema_200"] - 1

# ==========================================
# MOMENTUM
# ==========================================

delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df["rsi_14"] = 100 - (100 / (1 + rs))

bb_mid = df["close"].rolling(20).mean()
bb_std = df["close"].rolling(20).std()
df["bb_z_20"] = (df["close"] - bb_mid) / bb_std

# ==========================================
# RISK
# ==========================================

tr = pd.concat(
    [
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ],
    axis=1,
).max(axis=1)

df["atr_14"]  = tr.rolling(14).mean()
df["atrp_14"] = df["atr_14"] / df["close"]

df["dd_63"] = df["close"] / df["close"].rolling(63).max() - 1
df["dist_252_high"] = df["close"] / df["close"].rolling(252).max() - 1

# ==========================================
# FORWARD RETURN (5D TARGET)
# ==========================================

df["fwd_ret"] = np.log(df["close"].shift(-5) / df["close"])

# ==========================================
# TEMP REGIME LABEL (PIPELINE CHECK)
# ==========================================

df["regime"] = "calm"
df.loc[df["vol_20"] > 0.025, "regime"] = "volatile_event"
df.loc[df["dd_63"] < -0.20, "regime"] = "crisis"
df.loc[
    (df["regime"] == "calm") &
    (df["trend_20_50"].abs() > 0.02),
    "regime"
] = "trend_risk"

# ==========================================
# FINALIZE
# ==========================================

df = df.dropna()
df = df.reset_index().rename(columns={"Date": "date"})

df.to_csv("qqq_features_with_regime.csv", index=False)

print("Dataset created: qqq_features_with_regime.csv")
print(df.head())

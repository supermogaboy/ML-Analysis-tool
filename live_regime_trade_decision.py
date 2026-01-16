import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# =========================================================
# USER CONTROLS
# =========================================================

TICKER = "QQQ"
LOOKBACK_DAYS = 180          # 180 = 6 months, 365 = 1 year
EV_THRESHOLD = 0.0           # buy only if EV > threshold
MODEL_PATH = "regime_conditional_models.pkl"

# =========================================================
# FEATURES (MUST MATCH TRAINING)
# =========================================================

FEATURES = [
    "logret_1", "logret_5", "logret_10",
    "vol_10", "vol_20", "vol_z20",
    "ema_20", "ema_50", "ema_200",
    "trend_20_50", "trend_px_200",
    "rsi_14", "bb_z_20",
    "atr_14", "atrp_14",
    "dd_63", "dist_252_high"
]

# =========================================================
# DOWNLOAD RECENT DATA
# =========================================================

end_date = datetime.today()
start_date = end_date - timedelta(days=LOOKBACK_DAYS + 300)

df = yf.download(TICKER, start=start_date, end=end_date)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = df.columns.str.lower()
df = df[["open", "high", "low", "close", "volume"]].dropna()

# =========================================================
# FEATURE ENGINEERING (MATCH TRAINING)
# =========================================================

df["logret_1"]  = np.log(df["close"] / df["close"].shift(1))
df["logret_5"]  = np.log(df["close"] / df["close"].shift(5))
df["logret_10"] = np.log(df["close"] / df["close"].shift(10))

df["vol_10"] = df["logret_1"].rolling(10).std()
df["vol_20"] = df["logret_1"].rolling(20).std()
df["vol_z20"] = (
    df["vol_20"] - df["vol_20"].rolling(252).mean()
) / df["vol_20"].rolling(252).std()

df["ema_20"]  = df["close"].ewm(span=20).mean()
df["ema_50"]  = df["close"].ewm(span=50).mean()
df["ema_200"] = df["close"].ewm(span=200).mean()

df["trend_20_50"]  = df["ema_20"] / df["ema_50"] - 1
df["trend_px_200"] = df["close"] / df["ema_200"] - 1

delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df["rsi_14"] = 100 - (100 / (1 + rs))

bb_mid = df["close"].rolling(20).mean()
bb_std = df["close"].rolling(20).std()
df["bb_z_20"] = (df["close"] - bb_mid) / bb_std

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

df = df.dropna()

# =========================================================
# REGIME CLASSIFICATION (RULE-BASED, ROBUST)
# =========================================================

recent = df.tail(LOOKBACK_DAYS)

avg_vol = recent["vol_20"].mean()
max_dd = recent["dd_63"].min()
trend_strength = recent["trend_20_50"].mean()

if max_dd < -0.25:
    regime = "crisis"
elif avg_vol > 0.025:
    regime = "volatile_event"
elif avg_vol < 0.012 and abs(trend_strength) < 0.01:
    regime = "calm"
else:
    regime = "trend_risk"

# =========================================================
# LOAD REGIME-CONDITIONAL MODELS
# =========================================================

models = joblib.load(MODEL_PATH)

if regime not in models:
    raise ValueError(f"No model bundle found for regime: {regime}")

m = models[regime]

# =========================================================
# MODEL INFERENCE
# =========================================================

latest = df.iloc[-1]
X = latest[FEATURES].values.reshape(1, -1)
Xs = m["scaler"].transform(X)

p_up = m["p_up"].predict_proba(Xs)[0, 1]
p_down = 1 - p_up

ret_up = m["ret_up"].predict(Xs)[0]
ret_down = m["ret_down"].predict(Xs)[0]

EV_5D = p_up * ret_up + p_down * ret_down

# =========================================================
# DECISION
# =========================================================

good_trade = EV_5D > EV_THRESHOLD

# =========================================================
# OUTPUT (VERBOSE, EXPLAINABLE)
# =========================================================

print("\n================ LIVE REGIME TRADE DECISION ================\n")

print(f"Ticker:                {TICKER}")
print(f"Evaluation Date:       {df.index[-1].date()}")
print(f"Lookback Window:       {LOOKBACK_DAYS} days\n")

print("---- REGIME DIAGNOSTICS ----")
print(f"Avg 20D Volatility:    {avg_vol:.4f}")
print(f"Max 63D Drawdown:      {max_dd:.3f}")
print(f"Trend (20/50 EMA):     {trend_strength:.4f}")
print(f"➡ Detected Regime:     {regime}\n")

print("---- MODEL BUNDLE USED ----")
print(f"Probability Model:     LogisticRegression (p_up)")
print(f"Return Models:         GradientBoosting (up/down)\n")

print("---- MODEL INFERENCE (5D HORIZON) ----")
print(f"P(Up):                 {p_up:.3f}")
print(f"P(Down):               {p_down:.3f}")
print(f"Expected Return (Up):  {ret_up:.4f}")
print(f"Expected Return (Dn):  {ret_down:.4f}")
print(f"Expected Value (5D):   {EV_5D:.4f}\n")
EV_5D = p_up * ret_up + p_down * ret_down

current_price = latest["close"]
expected_price_5d = current_price * np.exp(EV_5D)

print("---- PRICE LEVELS ----")
print(f"Current Price:         ${current_price:.2f}")
print(f"Expected Price (5D):   ${expected_price_5d:.2f}")
print(f"Expected % Move (5D):  {(expected_price_5d/current_price - 1)*100:.2f}%\n")


print("=============== FINAL VERDICT =================")
if good_trade:
    print("✅ GOOD TRADE — BUY SIGNAL")
else:
    print("❌ BAD TRADE — NO BUY")

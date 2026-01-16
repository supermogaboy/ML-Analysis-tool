import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

# =========================================================
# USER CONTROLS
# =========================================================

DATA_PATH = "qqq_features_with_regime.csv"
MODEL_OUT = "regime_conditional_models.pkl"

# ---------------------------------
# PER-REGIME TRAINING WINDOWS
# ---------------------------------
REGIME_TRAIN_WINDOWS = {
    "calm": {
        "start": "2000-01-01",
        "end":   "2024-12-31"
    },
    "trend_risk": {
        "start": "2000-01-01",
        "end":   "2024-12-31"
    },
    "volatile_event": {
        "start": "2000-01-01",
        "end":   "2024-12-31"
    },
    "crisis": {
        "start": "2000-01-01",
        "end":   "2024-12-31"
    }
}

# =========================================================
# FEATURE SELECTION (NO LEAKAGE)
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
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

required_cols = FEATURES + ["fwd_ret", "regime"]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

models = {}

# =========================================================
# TRAIN REGIME-CONDITIONAL MODELS
# =========================================================

for regime, window in REGIME_TRAIN_WINDOWS.items():
    print(f"\nTraining regime: {regime}")

    start = window["start"]
    end = window["end"]

    df_r = df[
        (df["regime"] == regime) &
        (df.index >= start) &
        (df.index <= end)
    ].dropna(subset=required_cols)

    if len(df_r) < 200:
        print("  Skipped (not enough samples)")
        continue

    # Targets
    y_up = (df_r["fwd_ret"] > 0).astype(int)
    y_down = (df_r["fwd_ret"] < 0).astype(int)

    if y_up.sum() < 50 or y_down.sum() < 50:
        print("  Skipped (insufficient up/down samples)")
        continue

    # Features
    X = df_r[FEATURES]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---------------------------------
    # Probability Models
    # ---------------------------------
    p_up_model = LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    )
    p_down_model = LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    )

    p_up_model.fit(Xs, y_up)
    p_down_model.fit(Xs, y_down)

    # ---------------------------------
    # Return Magnitude Models (5D)
    # ---------------------------------
    ret_up_model = GradientBoostingRegressor()
    ret_down_model = GradientBoostingRegressor()

    ret_up_model.fit(
        Xs[y_up == 1],
        df_r.loc[y_up == 1, "fwd_ret"]
    )

    ret_down_model.fit(
        Xs[y_down == 1],
        df_r.loc[y_down == 1, "fwd_ret"]
    )

    # ---------------------------------
    # Store
    # ---------------------------------
    models[regime] = {
        "train_start": start,
        "train_end": end,
        "scaler": scaler,
        "p_up": p_up_model,
        "p_down": p_down_model,
        "ret_up": ret_up_model,
        "ret_down": ret_down_model,
        "features": FEATURES
    }

    print(f"  Trained on {len(df_r)} samples")

# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(models, MODEL_OUT)

print("\nTraining complete.")
print(f"Saved {len(models) * 4} models to: {MODEL_OUT}")

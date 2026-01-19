import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from config_loader import load_config, get_paths, get_features, get_training_params

# =========================================================
# LOAD CONFIGURATION
# =========================================================

config = load_config()
paths = get_paths()
features = get_features()
training_params = get_training_params()

DATA_PATH = f"{paths['processed_dir']}/QQQ_hmm_enhanced.csv"
MODEL_OUT = paths["regime_models"]
REGIME_TRAIN_WINDOWS = training_params["train_windows"]

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

required_cols = features + ["fwd_ret", "regime"]
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
    X = df_r[features]

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
        "features": features
    }

    print(f"  Trained on {len(df_r)} samples")

# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(models, MODEL_OUT)

print("\nTraining complete.")
print(f"Saved {len(models) * 4} models to: {MODEL_OUT}")

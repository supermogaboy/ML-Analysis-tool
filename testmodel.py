import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================================================
# USER CONTROLS
# =========================================================

DATA_PATH = "qqq_features_with_regime.csv"
MODEL_PATH = "regime_conditional_models.pkl"

TEST_START = "2000-01-01"
TEST_END   = "2026-12-31"

PROB_THRESHOLD = 0.5   # for direction decision

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
# LOAD DATA + MODELS
# =========================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

models = joblib.load(MODEL_PATH)

# Restrict test window
df_test = df.loc[TEST_START:TEST_END].dropna(subset=FEATURES + ["fwd_ret", "regime"])

# =========================================================
# STORAGE
# =========================================================

records = []

# =========================================================
# MAIN EVALUATION LOOP
# =========================================================

for date, row in df_test.iterrows():
    regime = row["regime"]

    if regime not in models:
        continue

    m = models[regime]

    X = row[FEATURES].values.reshape(1, -1)
    Xs = m["scaler"].transform(X)

    # Probabilities
    p_up = m["p_up"].predict_proba(Xs)[0, 1]
    p_down = m["p_down"].predict_proba(Xs)[0, 1]

    # Direction prediction
    pred_up = p_up >= PROB_THRESHOLD

    # True outcome
    true_up = row["fwd_ret"] > 0

    records.append({
        "date": date,
        "regime": regime,
        "p_up": p_up,
        "p_down": p_down,
        "pred_up": pred_up,
        "true_up": true_up,
        "fwd_ret": row["fwd_ret"]
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

res = pd.DataFrame(records).set_index("date")

# =========================================================
# METRICS
# =========================================================

overall_accuracy = accuracy_score(res["true_up"], res["pred_up"])
overall_auc = roc_auc_score(res["true_up"], res["p_up"])

print("\n================ OVERALL RESULTS ================")
print(f"Accuracy (direction, 5D): {overall_accuracy:.3f}")
print(f"AUC (P_up vs true):       {overall_auc:.3f}")

print("\n================ BY REGIME ======================")

for regime in sorted(res["regime"].unique()):
    r = res[res["regime"] == regime]

    if len(r) < 50:
        continue

    acc = accuracy_score(r["true_up"], r["pred_up"])
    auc = roc_auc_score(r["true_up"], r["p_up"])

    print(f"\nRegime: {regime}")
    print(f"  Samples: {len(r)}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUC:      {auc:.3f}")

# =========================================================
# BASELINE COMPARISON
# =========================================================

baseline_acc = accuracy_score(res["true_up"], np.ones(len(res), dtype=bool))

print("\n================ BASELINE ========================")
print(f"Always-Up Accuracy: {baseline_acc:.3f}")

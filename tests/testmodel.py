import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_loader import load_config, get_paths, get_features, get_testing_params

# =========================================================
# LOAD CONFIGURATION
# =========================================================

config = load_config()
paths = get_paths()
features = get_features()
testing_params = get_testing_params()

DATA_PATH = f"{paths['processed_dir']}/QQQ_hmm_enhanced.csv"
MODEL_PATH = paths["regime_models"]
TEST_START = testing_params["test_start"]
TEST_END = testing_params["test_end"]

# =========================================================
# LOAD DATA + MODELS
# =========================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

models = joblib.load(MODEL_PATH)

# Restrict test window
df_test = df.loc[TEST_START:TEST_END].dropna(subset=features + ["fwd_ret", "regime"])

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

    X = pd.DataFrame([row[features].values], columns=features)
    Xs = m["scaler"].transform(X)

    # Probabilities
    p_up = m["p_up"].predict_proba(Xs)[0, 1]
    p_down = m["p_down"].predict_proba(Xs)[0, 1]
    
    # Return magnitude predictions
    ret_up = m["ret_up"].predict(Xs)[0]
    ret_down = m["ret_down"].predict(Xs)[0]
    
    # True outcome
    true_up = row["fwd_ret"] > 0
    actual_ret = row["fwd_ret"]

    records.append({
        "date": date,
        "regime": regime,
        "p_up": p_up,
        "p_down": p_down,
        "ret_up": ret_up,
        "ret_down": ret_down,
        "true_up": true_up,
        "actual_ret": actual_ret,
        "fwd_ret": row["fwd_ret"]
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

res = pd.DataFrame(records).set_index("date")

# =========================================================
# DETAILED RESULTS
# =========================================================

print(f"\n📊 DETAILED PREDICTIONS ({len(res)} samples)")
print("=" * 100)
print(f"{'Date':<12} {'Regime':<15} {'P_Up':<7} {'P_Down':<7} {'Ret_Up':<8} {'Ret_Down':<8} {'Actual':<8} {'True_Up':<7}")
print("-" * 100)

for date, row in res.iterrows():
    print(f"{date.strftime('%Y-%m-%d'):<12} {row['regime']:<15} "
          f"{row['p_up']:.3f}   {row['p_down']:.3f}   "
          f"{row['ret_up']:.3f}   {row['ret_down']:.3f}   "
          f"{row['actual_ret']:.3f}   {row['true_up']}")

# =========================================================
# STATISTICS BY REGIME
# =========================================================

print(f"\n📈 STATISTICS BY REGIME")
print("=" * 80)

for regime in sorted(res["regime"].unique()):
    r = res[res["regime"] == regime]
    
    if len(r) < 10:
        print(f"\n{regime}: {len(r)} samples (too few for meaningful stats)")
        continue
    
    print(f"\n{regime.upper()} ({len(r)} samples)")
    print("-" * 40)
    
    # Probability stats
    print(f"P_Up:    mean={r['p_up'].mean():.3f}, std={r['p_up'].std():.3f}")
    print(f"P_Down:  mean={r['p_down'].mean():.3f}, std={r['p_down'].std():.3f}")
    
    # Return magnitude stats
    print(f"Ret_Up:  mean={r['ret_up'].mean():.3f}, std={r['ret_up'].std():.3f}")
    print(f"Ret_Down: mean={r['ret_down'].mean():.3f}, std={r['ret_down'].std():.3f}")
    
    # Actual returns
    print(f"Actual:  mean={r['actual_ret'].mean():.3f}, std={r['actual_ret'].std():.3f}")
    
    # Direction accuracy (using probability > 0.5)
    pred_up = r['p_up'] > 0.5
    accuracy = (pred_up == r['true_up']).mean()
    print(f"Accuracy: {accuracy:.3f}")

# =========================================================
# OVERALL STATISTICS
# =========================================================

print(f"\n🎯 OVERALL STATISTICS")
print("=" * 50)
print(f"Total samples: {len(res)}")
print(f"P_Up average: {res['p_up'].mean():.3f} (std: {res['p_up'].std():.3f})")
print(f"P_Down average: {res['p_down'].mean():.3f} (std: {res['p_down'].std():.3f})")
print(f"Actual return average: {res['actual_ret'].mean():.3f} (std: {res['actual_ret'].std():.3f})")

# Overall accuracy
pred_up = res['p_up'] > 0.5
overall_accuracy = (pred_up == res['true_up']).mean()
print(f"Overall accuracy: {overall_accuracy:.3f}")

# Save detailed results
res.to_csv("detailed_test_results.csv")
print(f"\n💾 Detailed results saved to: detailed_test_results.csv")

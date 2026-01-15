"""
Calm Regime Specialist – Inference / Scoring
===========================================

This module loads the trained Calm specialist models produced by
`Mode_train/Calm_Model.py` and runs them on QQQ daily data.

Inference rules
---------------
- Uses ONLY past prices (no forward labels or future prices).
- Computes the same feature set as in training for consistency.
- Applies the Calm regime filter at inference using `CalmRegimeConfig`.
- Scores all dates with valid rolling features; non‑calm rows are kept
  but their predictions are NaN.

Outputs for each scored date:
    - calm (bool)
    - P(down), P(flat), P(up)
    - Expected upside magnitude U (forward log‑return, conditional positive)
    - Expected downside magnitude D (absolute value, conditional negative)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import joblib
import pandas as pd

# Ensure project root is on sys.path so we can import the training module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Mode_train.Calm_Model import (  # type: ignore[import]
    CalmRegimeConfig,
    compute_features,
    identify_calm_regime,
    load_qqq_daily_csv,
)


def _load_trained_models(
    model_dir: str,
) -> Tuple[object, object, object, Dict]:
    """
    Load trained Calm specialist models and metadata from disk.
    """
    directional_path = os.path.join(model_dir, "calm_directional.pkl")
    upside_path = os.path.join(model_dir, "calm_upside.pkl")
    downside_path = os.path.join(model_dir, "calm_downside.pkl")
    metadata_path = os.path.join(model_dir, "calm_metadata.json")

    if not os.path.exists(directional_path):
        raise FileNotFoundError(
            f"Directional model not found at {directional_path}. "
            "Run Calm_Model.py training first."
        )
    if not os.path.exists(upside_path) or not os.path.exists(downside_path):
        raise FileNotFoundError(
            "Magnitude models not found. Run Calm_Model.py training first."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Metadata file calm_metadata.json not found. "
            "Ensure Calm_Model.py training finished successfully."
        )

    directional_model = joblib.load(directional_path)
    upside_model = joblib.load(upside_path)
    downside_model = joblib.load(downside_path)

    metadata = pd.read_json(metadata_path, typ="series").to_dict()

    return directional_model, upside_model, downside_model, metadata


def score_calm_specialist_on_dataframe(
    df: pd.DataFrame,
    model_dir: str = "Model",
    config: CalmRegimeConfig | None = None,
) -> pd.DataFrame:
    """
    Run the Calm specialist on a raw QQQ daily DataFrame.

    This does NOT re‑train the model – it uses the saved weights/structure
    produced earlier by `Calm_Model.py`.

    Parameters
    ----------
    df : DataFrame
        Raw OHLCV data (same structure as training CSV).
    model_dir : str
        Directory containing the saved calm_* model files and metadata.
    config : CalmRegimeConfig, optional
        If None, uses default configuration (must match training horizon/flat band).

    Returns
    -------
    DataFrame
        Indexed by date, with columns:
            - p_down, p_flat, p_up
            - exp_upside, exp_downside
    """
    if config is None:
        config = CalmRegimeConfig()

    # Basic input validation
    required_cols = {"date", "adj_close", "close"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")

    # Ensure sorted by date ascending before feature calculation
    df = df.sort_values("date").reset_index(drop=True)

    (
        directional_model,
        upside_model,
        downside_model,
        metadata,
    ) = _load_trained_models(model_dir=model_dir)

    # Recreate feature set exactly as in training (no labels / forward returns)
    features = compute_features(df)

    feature_names = metadata.get("feature_names")
    if feature_names is None:
        raise ValueError("Missing 'feature_names' in calm_metadata.json.")

    # Restrict to rows where all features are available (rolling windows matured)
    feature_frame = features.set_index("date")
    X_all = feature_frame[feature_names]
    feature_valid_mask = ~X_all.isna().any(axis=1)

    # Apply Calm regime filter on the same feature set
    calm_mask = pd.Series(
        identify_calm_regime(
            features=feature_frame.loc[feature_valid_mask].reset_index(),
            config=config,
        ),
        index=feature_frame.index[feature_valid_mask],
    )

    # Base output DataFrame with calm flag; NaN calm where features invalid
    out = pd.DataFrame(index=feature_frame.index)
    out["calm"] = False
    out.loc[calm_mask.index, "calm"] = calm_mask.values

    # Initialise prediction columns as NaN
    for col in ["p_down", "p_flat", "p_up", "exp_upside", "exp_downside"]:
        out[col] = pd.Series(index=out.index, dtype="float64")

    # Only score rows that:
    #   - have all required features (feature_valid_mask)
    #   - are in the Calm regime (calm == True)
    score_idx = out.index[feature_valid_mask & out["calm"]]
    if len(score_idx) == 0:
        # No rows to score; return calm flags + NaN predictions
        return out

    X = X_all.loc[score_idx]

    # Directional probabilities (keep .values)
    prob = directional_model.predict_proba(X.values)
    p_down = prob[:, 0]
    p_flat = prob[:, 1]
    p_up = prob[:, 2]

    # Magnitudes (keep .values)
    exp_upside = upside_model.predict(X.values)
    exp_downside = downside_model.predict(X.values)

    out.loc[score_idx, "p_down"] = p_down
    out.loc[score_idx, "p_flat"] = p_flat
    out.loc[score_idx, "p_up"] = p_up
    out.loc[score_idx, "exp_upside"] = exp_upside
    out.loc[score_idx, "exp_downside"] = exp_downside

    out.index.name = "date"
    return out


def score_calm_specialist(
    csv_path: str = "data/raw/QQQ_1d.csv",
    model_dir: str = "Model",
    config: CalmRegimeConfig | None = None,
) -> Dict[str, float]:
    """
    Convenience wrapper:
    - loads latest QQQ CSV
    - loads Calm specialist models
    - scores all eligible dates

    Returns a single dictionary for the **most recent date** the model can
    score (i.e. where features + forward horizon are fully defined).
    """
    if config is None:
        config = CalmRegimeConfig()

    df = load_qqq_daily_csv(csv_path)
    scores = score_calm_specialist_on_dataframe(
        df=df,
        model_dir=model_dir,
        config=config,
    )

    if scores.empty:
        raise ValueError("No eligible rows to score. Check input data length.")

    last_row = scores.iloc[-1]
    result = {
        "date": str(scores.index[-1]),
        "calm": bool(last_row["calm"]),
        "p_down": float(last_row["p_down"]) if pd.notna(last_row["p_down"]) else float("nan"),
        "p_flat": float(last_row["p_flat"]) if pd.notna(last_row["p_flat"]) else float("nan"),
        "p_up": float(last_row["p_up"]) if pd.notna(last_row["p_up"]) else float("nan"),
        "exp_upside": float(last_row["exp_upside"]) if pd.notna(last_row["exp_upside"]) else float(
            "nan"
        ),
        "exp_downside": float(last_row["exp_downside"]) if pd.notna(
            last_row["exp_downside"]
        ) else float("nan"),
    }

    return result


if __name__ == "__main__":
    # Example CLI usage
    res = score_calm_specialist()
    print("Latest Calm specialist scores:")
    for k, v in res.items():
        print(f"  {k}: {v}")

co
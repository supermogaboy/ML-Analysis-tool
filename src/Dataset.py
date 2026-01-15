# src/dataset.py
import numpy as np
import pandas as pd
from src.Config import HORIZON_DAYS, FLAT_BAND

def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
    - y_class in {0:down, 1:flat, 2:up} based on forward return
    - U = max(fwd_ret, 0)  (upside magnitude)
    - D = max(-fwd_ret, 0) (downside magnitude)
    """
    d = df.copy()
    fwd = d["close"].shift(-HORIZON_DAYS) / d["close"] - 1.0
    d["fwd_ret"] = fwd

    y = np.full(len(d), 1)  # flat default
    y[fwd > FLAT_BAND] = 2
    y[fwd < -FLAT_BAND] = 0
    d["y_class"] = y

    d["U"] = fwd.clip(lower=0.0)
    d["D"] = (-fwd).clip(lower=0.0)

    return d

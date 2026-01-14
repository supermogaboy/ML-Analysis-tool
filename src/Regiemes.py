# src/regimes.py
import numpy as np
import pandas as pd

REGIMES = ["calm", "trend_risk", "volatile_event", "crisis"]

def label_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Uses simple, explainable rules to label regimes.
    Requires features: vol_20, trend_20_50, vol_z20, dd_63, logret_1
    """
    d = df.copy()

    vol = d["vol_20"]
    vol_q90 = vol.quantile(0.90)
    vol_q25 = vol.quantile(0.25)

    abs_ret = d["logret_1"].abs()
    spike = abs_ret > 2.5 * (d["vol_20"] + 1e-12)

    crisis = (d["dd_63"] < -0.12) & (d["vol_20"] > vol_q90)

    volatile_event = (~crisis) & (spike | (d["vol_z20"] > 2.0))

    trend_risk = (~crisis) & (~volatile_event) & (
        d["trend_20_50"].abs() > d["trend_20_50"].abs().quantile(0.65)
    ) & (d["vol_20"] > vol_q25)

    calm = (~crisis) & (~volatile_event) & (~trend_risk)

    regime = pd.Series(index=d.index, dtype="object")
    regime[crisis] = "crisis"
    regime[volatile_event] = "volatile_event"
    regime[trend_risk] = "trend_risk"
    regime[calm] = "calm"

    # Fallback
    regime = regime.fillna("calm")
    return regime

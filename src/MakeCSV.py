# src/make_csv.py
import os
import pandas as pd
import yfinance as yf
from src.Config import SYMBOL, INTERVAL, START, END

def main():
    os.makedirs("data/raw", exist_ok=True)

    df = yf.download(
        SYMBOL,
        start=START,
        end=END,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("No data returned. Check symbol/interval/start/end.")

    df = df.reset_index()
    # For single ticker, yfinance returns MultiIndex columns; flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Expected columns for 1d: date, open, high, low, close, adj_close, volume
    out = f"data/raw/{SYMBOL}_{INTERVAL}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} with {len(df)} rows")

if __name__ == "__main__":
    main()

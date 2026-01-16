import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from RegimeClassifier import RegimeClassifier

# -----------------------------
# Download QQQ Data
# -----------------------------
df = yf.download("QQQ", start="2000-01-01", end="2025-12-31", auto_adjust=False)
# Handle MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df[["Adj Close"]].dropna()

# -----------------------------
# Train on 2000–2024
# -----------------------------
train_df = df.loc[: "2024-12-31"]
test_df = df.loc["2025-01-01":]

model = RegimeClassifier()
model.fit(train_df)

# Save model for reuse
model.save("qqq_regime_model.pkl")

# -----------------------------
# Predict regimes
# -----------------------------
train_regimes = model.predict(train_df)
test_regimes = model.predict(test_df)

df_plot = df.loc[train_regimes.index.union(test_regimes.index)]
df_plot["regime"] = train_regimes.combine_first(test_regimes)

# -----------------------------
# Plot
# -----------------------------
colors = {
    "Calm": "green",
    "General Trend with Risk": "blue",
    "Volatile Event": "orange",
    "Crisis": "red"
}

plt.figure(figsize=(15,6))
plt.plot(df_plot["Adj Close"], color="black", alpha=0.6)

for regime, color in colors.items():
    mask = df_plot["regime"] == regime
    plt.scatter(
        df_plot.index[mask],
        df_plot["Adj Close"][mask],
        s=6,
        label=regime,
        color=color
    )

plt.legend()
plt.title("QQQ Market Regimes (Unsupervised, GMM)")
plt.show()


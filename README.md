# ML Trading Regime Models

Minimal viable research pipeline for training regime-specialist models with directional forecasts.

## Overview

Trains 4 regime-specialist models (TREND, CHOP, CALM, CRISIS) that output:
- **Up-side forecast**: Expected positive return and confidence
- **Down-side forecast**: Expected negative return and confidence

Each regime uses separate up/down models following Option A architecture.

## Data Requirements

### Input CSV Format
Expected CSV file: `data/processed/QQQ_1d_processed.csv`

**Required columns:**
- `regime`: Regime label (trend/chop/calm/crisis)  
- `fwd_ret`: Forward return over horizon H
- Feature columns: Any technical indicators/features

**Example columns:**
```
date,adj_close,close,high,low,open,volume,logret_1,vol_10,ema_20,rsi_14,regime,fwd_ret
```

### Regime Filtering
- Data is filtered by `regime` column
- Each model trains only on data points belonging to its regime
- Regime labels must be lowercase: {trend, chop, calm, crisis}

## Model Architecture

### Up Models per Regime
- **Confidence model**: `GradientBoostingClassifier` predicts P(up)
- **Magnitude model**: `GradientBoostingRegressor` predicts E[r | r > 0]

### Down Models per Regime  
- **Confidence model**: `GradientBoostingClassifier` predicts P(down)
- **Magnitude model**: `GradientBoostingRegressor` predicts E[r | r < 0]

### Final Outputs
- `up.expected_return = P(up) × E[r | r > 0]` (always ≥ 0)
- `down.expected_return = P(down) × E[r | r < 0]` (always ≤ 0)
- `up.confidence = P(up)` in [0,1]
- `down.confidence = P(down)` in [0,1]

## Usage

### Data Processing
```bash
# Process raw data into features and labels
python run_data_pipeline.py
```

This will:
- Download QQQ data from yfinance (or use existing data/raw/QQQ_1d.csv)
- Create technical indicators and features
- **Train RegimeClassifier using Gaussian Mixture Models** for regime detection
- Detect market regimes (trend_risk, calm, crisis, volatile_event)
- Generate forward returns and labels
- Save to data/processed/QQQ_1d_processed.csv

**Regime Detection**: Uses unsupervised Gaussian Mixture Models to identify 4 market regimes based on volatility, trend, drawdown, and Sharpe ratio characteristics. The classifier is trained automatically and saved to `models/regime_classifier.pkl`.

### Training
```bash
python train_regime_models.py
```

**Outputs:**
```
models/
├── trend_risk/
│   ├── model_up_clf.pkl      # Up confidence model
│   ├── model_up_reg.pkl      # Up magnitude model  
│   ├── model_down_clf.pkl    # Down confidence model
│   ├── model_down_reg.pkl    # Down magnitude model
│   ├── feature_list.json     # Feature names/order
│   ├── config.json           # Training metadata
│   └── metrics.json          # Performance metrics
├── volatile_event/
├── calm/
├── crisis/
└── chop/

### Inference
```python
from infer import predict_regime
import pandas as pd

# Load your feature data (single row)
X_row = pd.DataFrame({
    'adj_close': [100.0],
    'close': [100.0], 
    'high': [105.0],
    'low': [95.0],
    'open': [98.0],
    'volume': [1000000],
    'logret_1': [0.01],
    'logret_5': [0.02],
    'logret_10': [0.03],
    'vol_10': [0.015],
    'vol_20': [0.018],
    'ema_20': [99.0],
    'ema_50': [98.0],
    'ema_200': [97.0],
    'trend_20_50': [0.01],
    'trend_px_200': [0.03],
    'rsi_14': [55.0],
    'bb_z_20': [0.5],
    'atr_14': [2.0],
    'atrp_14': [0.02],
    'vol_sma20': [0.017],
    'vol_std20': [0.005],
    'vol_z20': [0.3],
    'dd_63': [0.05],
    'dist_252_high': [0.1]
})

# Get predictions for a regime
result = predict_regime("trend_risk", X_row)
print(result)
```

**Output format:**
```json
{
  "regime": "TREND",
  "up": {
    "expected_return": 0.0234,
    "confidence": 0.67
  },
  "down": {
    "expected_return": -0.0156, 
    "confidence": 0.33
  }
}
```

### Utility Functions
```python
from infer import list_available_regimes, get_regime_info

# List trained regimes
regimes = list_available_regimes()

# Get regime metadata
info = get_regime_info("trend")
print(info["metrics"])  # AUC, MAE, RMSE
print(info["config"])   # Training date, horizon, etc.
```

## Configuration

Key parameters in `train_regime_models.py`:
- `HORIZON_DAYS = 5`: Forward return horizon
- `REGIMES = ["trend", "chop", "calm", "crisis"]`: Regime list
- `RANDOM_SEED = 42`: Reproducibility

## Dependencies

```
pandas>=2.0.0
numpy>=1.20.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

Install with:
```bash
pip install pandas numpy scikit-learn joblib
```

## Training Details

- **Time series split**: 80% train, 20% test (chronological)
- **Model**: GradientBoosting with 100 trees, max_depth=3
- **Handling insufficient data**: Creates dummy models if <10 samples
- **Metrics**: AUC for confidence models, MAE/RMSE for magnitude models

### Hackathon Demo
```bash
# Complete pipeline demonstration for judges
python demo.py
```

**Demo shows:**
- Dataset overview and regime distribution
- Model performance metrics across all regimes
- Live inference with realistic market data
- Performance comparison and feature importance
- Complete end-to-end pipeline validation

## File Structure
```
├── demo.py                   # Hackathon demo script
├── run_data_pipeline.py         # Data processing script
├── train_regime_models.py      # Training script
├── infer.py                   # Inference module
├── README.md                  # This file
├── requirements_minimal.txt     # Minimal dependencies
├── src/                       # Data processing modules
│   ├── Features.py            # Technical indicator calculations
│   ├── Regimes.py            # Regime detection logic (integrates RegimeClassifier)
│   ├── Dataset.py            # Data loading/processing utilities
│   └── Config.py            # Configuration settings
├── classifier/               # Regime classifier module
│   └── RegimeClassifier.py   # Gaussian Mixture Model regime detection
├── data/
│   ├── raw/                   # Raw OHLCV data
│   │   └── QQQ_1d.csv
│   └── processed/             # Features and labels
│       └── QQQ_1d_processed.csv
└── models/                   # Trained models (created after training)
    ├── regime_classifier.pkl  # Trained regime classifier
    ├── trend_risk/
    ├── volatile_event/
    ├── calm/
    ├── crisis/
    └── chop/
```

## Notes

- Models are trained independently per regime - no blending or ensemble
- Confidence scores are calibrated probabilities from GradientBoosting
- Expected returns combine probability and magnitude predictions
- Feature ordering must match training (saved in feature_list.json)
- No portfolio construction, risk management, or position sizing included



# ML Trading System

Organized machine learning trading system with regime detection and conditional modeling.

## Directory Structure

```
ML-Analysis-tool/
├── configs/              # Configuration files
│   └── config.json      # Main configuration
├── src/                  # Source code
│   ├── config_loader.py   # Configuration management
│   ├── train_regime_models2.py  # Model training
│   └── ...             # Other source modules
├── tests/                # Test scripts
│   └── testmodel.py     # Model testing and evaluation
├── data/                 # Data files
│   ├── raw/             # Raw OHLCV data
│   └── processed/       # Processed data with features
├── models/               # Trained models
├── main.py              # Main entry point
└── run_data_pipeline.py  # Data processing pipeline
```

## Usage

### Quick Start
```bash
# Process data and train HMM
python main.py pipeline

# Train regime models
python main.py train

# Test models
python main.py test
```

### Individual Scripts
```bash
# Data processing with HMM
python run_data_pipeline.py --train-hmm

# Train models
python src/train_regime_models2.py

# Test models
python tests/testmodel.py
```

## Configuration

All parameters are managed through `configs/config.json`:
- Data paths and date ranges
- Model parameters
- Feature lists
- Training windows
- Test periods

## Workflow

1. **Pipeline**: Downloads data, creates features, trains HMM
2. **Train**: Trains 16 regime-conditional models (4 per regime)
3. **Test**: Evaluates models on unseen data with detailed output

## Models

- **HMM Regime Classifier**: Unsupervised regime detection
- **Direction Models**: P(Up) and P(Down) for each regime
- **Magnitude Models**: Expected returns for up/down moves per regime

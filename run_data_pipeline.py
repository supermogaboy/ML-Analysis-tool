#!/usr/bin/env python3

"""
Enhanced Data Processing Pipeline for ML Trading System with HMM Integration
========================================================================

This script processes raw OHLCV data into features and labels for ML training.
It downloads data, creates technical indicators, detects regimes, trains HMM,
and saves processed data for individual model training.

Flow:
1. Download data
2. Process data for features  
3. Label data (forward returns)
4. Train HMM regime classifier (unsupervised - no pre-labels needed!)
5. Save enhanced data with both regime types
6. Ready for individual model training

Usage:
    python run_data_pipeline.py [--train-hmm] [--hmm-regimes 4]
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "classifier"))

from src.Dataset import load_raw_data, process_data, save_processed_data, validate_data, summarize_dataset
from src.config_loader import load_config, get_paths, get_training_params
from classifier.HMMRegimeClassifier import HMMRegimeClassifier

def train_hmm_regime_classifier(df, n_regimes=4, model_save_path=None):
    """
    Train HMM regime classifier on processed data
    
    Args:
        df: Processed DataFrame with features
        n_regimes: Number of regimes for HMM
        model_save_path: Path to save HMM model
    
    Returns:
        Enhanced DataFrame with HMM regimes
    """
    config = load_config()
    training_params = get_training_params()
    
    if model_save_path is None:
        model_save_path = get_paths()["hmm_model"]
    
    print(f"\n🤖 Training HMM Regime Classifier")
    print("-" * 40)
    
    # Initialize HMM classifier
    hmm_classifier = HMMRegimeClassifier(
        n_regimes=n_regimes, 
        random_state=training_params["random_seed"]
    )
    
    # Train HMM (unsupervised - learns regimes from data patterns)
    print(f"📊 Training HMM with {n_regimes} regimes...")
    hmm_classifier.fit(df, price_col='close')
    
    # Get HMM predictions and probabilities
    df['hmm_regime'] = hmm_classifier.predict(df)
    regime_probs = hmm_classifier.predict_proba(df)
    
    # Add regime probabilities to DataFrame
    for col in regime_probs.columns:
        df[f'hmm_prob_{col}'] = regime_probs[col]
    
    # Analyze and display results
    print(f"\n📈 HMM Regime Distribution:")
    regime_counts = df['hmm_regime'].value_counts()
    for regime, count in regime_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {regime:15}: {count:4,} samples ({percentage:5.1f}%)")
    
    print(f"\n🔄 Transition Matrix:")
    trans_matrix = hmm_classifier.get_transition_matrix()
    print(trans_matrix.round(4))
    
    # Save HMM model
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    hmm_classifier.save(model_save_path)
    print(f"\n💾 HMM model saved to: {model_save_path}")
    
    return df, hmm_classifier


def main():
    parser = argparse.ArgumentParser(description="Enhanced Data Processing Pipeline with HMM")
    parser.add_argument("--train-hmm", action="store_true", 
                       help="Train HMM regime classifier")
    parser.add_argument("--hmm-regimes", type=int, default=4,
                       help="Number of regimes for HMM (default: 4)")
    parser.add_argument("--symbol", default=None,
                       help="Symbol to process (from config)")
    parser.add_argument("--start-date", default=None,
                       help="Start date (from config)")
    parser.add_argument("--end-date", default=None,
                       help="End date (from config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    paths = get_paths()
    training_params = get_training_params()
    
    # Use command line args or config defaults
    symbol = args.symbol or config["data"]["symbol"]
    start_date = args.start_date or config["data"]["start_date"]
    end_date = args.end_date or config["data"]["end_date"]
    horizon_days = training_params["horizon_days"]
    flat_band = training_params["flat_band"]
    random_seed = training_params["random_seed"]
    
    print("🚀 Starting Enhanced Data Processing Pipeline")
    print("=" * 50)
    
    # File paths
    raw_path = f"{paths['raw_dir']}/{symbol}_1d.csv"
    processed_path = f"{paths['processed_dir']}/{symbol}_1d_processed.csv"
    hmm_enhanced_path = f"{paths['processed_dir']}/{symbol}_hmm_enhanced.csv"
    
    # Step 1: Load or download raw data
    print(f"📥 Loading data for {symbol}...")
    
    # Try to load existing raw data first
    if os.path.exists(raw_path):
        print(f"Found existing raw data at {raw_path}")
        raw_df = pd.read_csv(raw_path)
        print(f"Loaded {len(raw_df)} rows of raw data")
    else:
        print(f"Downloading data from yfinance...")
        raw_df = load_raw_data(symbol, start_date, end_date)
        
        if raw_df is None:
            print("❌ Failed to load data. Exiting.")
            return
        
        # Handle MultiIndex columns from yfinance
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.droplevel(1)
        
        # Convert column names to lowercase to match expected format
        raw_df.columns = [col.lower() for col in raw_df.columns]
        
        # Save raw data
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        print(f"💾 Saved raw data to {raw_path}")
    
    # Step 2: Process data with basic features and labels
    print(f"\n⚙️ Processing data with horizon={horizon_days} days...")
    
    # Process WITHOUT existing regime classifier to avoid conflicts
    processed_df = process_data(
        raw_df, 
        horizon_days=horizon_days, 
        flat_band=flat_band,
        use_regime_classifier=False  # We'll add HMM regimes next
    )
    
    # Step 3: Validate processed data
    print(f"\n🔍 Validating processed data...")
    issues = validate_data(processed_df)
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Data validation passed")
    
    # Step 4: Train HMM if requested
    if args.train_hmm:
        enhanced_df, hmm_classifier = train_hmm_regime_classifier(
            processed_df, 
            n_regimes=args.hmm_regimes,
            model_save_path=paths["hmm_model"]
        )
        
        # Save enhanced data with HMM features
        print(f"\n💾 Saving HMM-enhanced data...")
        save_processed_data(enhanced_df, hmm_enhanced_path)
        print(f"📁 HMM-enhanced data saved to: {hmm_enhanced_path}")
        
        # Show final summary
        print(f"\n📊 Final Enhanced Dataset Summary:")
        print(f"   Total samples: {len(enhanced_df):,}")
        print(f"   Total features: {len([col for col in enhanced_df.columns if col not in ['date', 'regime', 'hmm_regime', 'fwd_ret', 'y_class', 'U', 'D']])}")
        print(f"   Regime types: Original (regime) + HMM (hmm_regime)")
        print(f"   HMM probabilities: {[col for col in enhanced_df.columns if col.startswith('hmm_prob_')]}")
        
    else:
        # Save basic processed data
        print(f"\n💾 Saving processed data...")
        save_processed_data(processed_df, processed_path)
        print(f"📁 Processed data saved to: {processed_path}")
        
        # Show basic summary
        print(f"\n📊 Dataset Summary:")
        summarize_dataset(processed_df)
    
    print(f"\n🎉 Data processing complete!")
    
    if args.train_hmm:
        print(f"📈 Ready for ML training with HMM features:")
        print(f"   python train_regime_models2.py  # Uses enhanced data")
        print(f"   python demo2.py               # Demo with HMM integration")
    else:
        print(f"📈 Ready for ML training:")
        print(f"   python train_regime_models2.py")
        print(f"   To add HMM: python run_data_pipeline.py --train-hmm")


if __name__ == "__main__":
    main()

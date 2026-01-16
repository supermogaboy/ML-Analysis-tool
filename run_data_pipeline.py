#!/usr/bin/env python3

"""
Data Processing Pipeline for ML Trading System
============================================

This script processes raw OHLCV data into features and labels for ML training.
It downloads data, creates technical indicators, detects regimes, and saves processed data.

Usage:
    python run_data_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.Dataset import load_raw_data, process_data, save_processed_data, validate_data, summarize_dataset
from src.Config import SYMBOL, START, END, HORIZON_DAYS, FLAT_BAND

def main():
    print("🚀 Starting Data Processing Pipeline")
    print("=" * 50)
    
    # File paths
    raw_path = f"data/raw/{SYMBOL}_1d.csv"
    processed_path = f"data/processed/{SYMBOL}_1d_processed.csv"
    
    # Step 1: Load or download raw data
    print(f"📥 Loading data for {SYMBOL}...")
    
    # Try to load existing raw data first
    if os.path.exists(raw_path):
        print(f"Found existing raw data at {raw_path}")
        import pandas as pd
        raw_df = pd.read_csv(raw_path)
        print(f"Loaded {len(raw_df)} rows of raw data")
    else:
        print(f"Downloading data from yfinance...")
        raw_df = load_raw_data(SYMBOL, START, END)
        
        if raw_df is None:
            print("❌ Failed to load data. Exiting.")
            return
        
        # Save raw data
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        print(f"💾 Saved raw data to {raw_path}")
    
    # Step 2: Process data
    print(f"\n⚙️ Processing data with horizon={HORIZON_DAYS} days...")
    
    processed_df = process_data(
        raw_df, 
        horizon_days=HORIZON_DAYS, 
        flat_band=FLAT_BAND
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
    
    # Step 4: Show summary
    print(f"\n📊 Dataset Summary:")
    summarize_dataset(processed_df)
    
    # Step 5: Save processed data
    print(f"\n💾 Saving processed data...")
    save_processed_data(processed_df, processed_path)
    
    print(f"\n🎉 Data processing complete!")
    print(f"📁 Processed data saved to: {processed_path}")
    print(f"📈 Ready for ML training with: python train_regime_models.py")

if __name__ == "__main__":
    main()

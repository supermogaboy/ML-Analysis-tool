#!/usr/bin/env python3
"""
Data Processing Pipeline
========================
Runs all src scripts in sequence to produce processed data:
1. Downloads raw data (MakeCSV.py)
2. Adds technical features (Features.py)
3. Creates target variables (Dataset.py)
4. Labels market regimes (Regiemes.py)

Output: Saves processed data to data/processed/
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Get project root (parent of src directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project root to path for imports (MakeCSV uses 'from src.Config import ...')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change to project root for relative paths
os.chdir(PROJECT_ROOT)

from src.Config import SYMBOL, INTERVAL, START, END
from src.MakeCSV import main as download_data
from src.Features import add_features
from src.Dataset import make_targets
from src.Regiemes import label_regimes


def run_pipeline(
    download_new_data: bool = False,
    input_csv: str = None,
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Run the complete data processing pipeline.
    
    Parameters
    ----------
    download_new_data : bool
        If True, download fresh data from yfinance. If False, use existing CSV.
    input_csv : str, optional
        Path to input CSV file. If None, uses default from Config.
    output_dir : str
        Directory to save processed data.
    
    Returns
    -------
    pd.DataFrame
        Processed dataframe with features, targets, and regime labels.
    """
    # Step 1: Download or load raw data
    if download_new_data:
        print("Step 1: Downloading raw data...")
        download_data()
        print("✓ Data downloaded successfully")
    
    # Determine input CSV path
    if input_csv is None:
        input_csv = f"data/raw/{SYMBOL}_{INTERVAL}.csv"
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Set download_new_data=True to download data first."
        )
    
    print(f"Step 1: Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} rows")
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Step 2: Add technical features
    print("\nStep 2: Adding technical features...")
    df = add_features(df)
    print(f"✓ Added features. DataFrame shape: {df.shape}")
    
    # Step 3: Create target variables
    print("\nStep 3: Creating target variables...")
    df = make_targets(df)
    print(f"✓ Created targets (y_class, U, D). DataFrame shape: {df.shape}")
    
    # Step 4: Label market regimes
    print("\nStep 4: Labeling market regimes...")
    df['regime'] = label_regimes(df)
    print(f"✓ Labeled regimes. DataFrame shape: {df.shape}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("Pipeline Summary")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nTarget class distribution:")
    print(df['y_class'].value_counts().sort_index())
    print(f"\nRegime distribution:")
    print(df['regime'].value_counts())
    print(f"\nFeatures: {len([c for c in df.columns if c not in ['date', 'y_class', 'U', 'D', 'regime', 'fwd_ret']])}")
    print("="*60)
    
    # Step 5: Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{SYMBOL}_{INTERVAL}_processed.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved processed data to: {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the complete data processing pipeline"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fresh data from yfinance"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file (default: data/raw/SYMBOL_INTERVAL.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data (default: data/processed)"
    )
    
    args = parser.parse_args()
    
    try:
        df = run_pipeline(
            download_new_data=args.download,
            input_csv=args.input,
            output_dir=args.output_dir
        )
        print("\n✓ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

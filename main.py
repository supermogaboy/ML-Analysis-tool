#!/usr/bin/env python3

"""
Main entry points for ML Trading System
"""

import subprocess
import sys
from pathlib import Path

def run_pipeline():
    """Run the complete data processing pipeline"""
    cmd = ["python", "run_data_pipeline.py", "--train-hmm"]
    subprocess.run(cmd, cwd=Path(__file__).parent)

def train_models():
    """Train regime-conditional models"""
    cmd = ["python", "src/train_regime_models2.py"]
    subprocess.run(cmd, cwd=Path(__file__).parent)

def test_models():
    """Test trained models"""
    cmd = ["python", "tests/testmodel.py"]
    subprocess.run(cmd, cwd=Path(__file__).parent)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py [pipeline|train|test]")
        return
    
    command = sys.argv[1]
    
    if command == "pipeline":
        run_pipeline()
    elif command == "train":
        train_models()
    elif command == "test":
        test_models()
    else:
        print(f"Unknown command: {command}")
        print("Available: pipeline, train, test")

if __name__ == "__main__":
    main()

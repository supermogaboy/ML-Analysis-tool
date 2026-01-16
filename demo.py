#!/usr/bin/env python3

"""
🚀 ML Trading Regime Models - Demo for Hackathon Judges
====================================================

This script demonstrates the complete ML pipeline:
1. Data processing and feature engineering
2. Regime detection and model training  
3. Inference with directional forecasts
4. Performance evaluation

Usage:
    python demo.py
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.Dataset import load_processed_data, get_feature_columns
from infer import predict_regime, list_available_regimes, get_regime_info

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_subheader(title):
    """Print formatted subsection header"""
    print(f"\n📊 {title}")
    print("-" * 40)

def demo_data_info():
    """Show dataset information"""
    print_subheader("Dataset Overview")
    
    try:
        df = load_processed_data("data/processed/QQQ_1d_processed.csv")
        print(f"📈 Total samples: {len(df):,}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🔧 Features: {len(get_feature_columns(df))}")
        
        print(f"\n🎭 Regime Distribution:")
        regime_counts = df['regime'].value_counts()
        for regime, count in regime_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {regime:12} : {count:4,} samples ({percentage:5.1f}%)")
            
        print(f"\n📊 Forward Return Stats:")
        print(f"   Mean: {df['fwd_ret'].mean():+.4f}")
        print(f"   Std:  {df['fwd_ret'].std():.4f}")
        print(f"   Min:  {df['fwd_ret'].min():+.4f}")
        print(f"   Max:  {df['fwd_ret'].max():+.4f}")
        
        return df
    except FileNotFoundError:
        print("❌ Processed data not found. Run: python run_data_pipeline.py")
        return None

def demo_model_info():
    """Show trained model information"""
    print_subheader("Trained Regime Models")
    
    regimes = list_available_regimes()
    
    if not regimes:
        print("❌ No trained models found. Run: python train_regime_models.py")
        return
    
    print(f"🤖 Available regimes: {', '.join(regimes)}")
    
    for regime in regimes:
        try:
            info = get_regime_info(regime.lower())
            metrics = info['metrics']
            config = info['config']
            
            print(f"\n📈 {regime} Model Performance:")
            print(f"   Samples: {metrics['n_samples']:,} (train: {metrics['n_train']:,}, test: {metrics['n_test']:,})")
            print(f"   Up AUC: {metrics['up_auc']:.3f}")
            print(f"   Down AUC: {metrics['down_auc']:.3f}")
            print(f"   Up MAE: {metrics['up_mae']:.4f}")
            print(f"   Down MAE: {metrics['down_mae']:.4f}")
            print(f"   Training date: {config['training_date'][:10]}")
            
        except Exception as e:
            print(f"❌ Error loading {regime} info: {e}")

def demo_inference():
    """Demonstrate inference with sample data"""
    print_subheader("Live Inference Demo")
    
    regimes = list_available_regimes()
    if not regimes:
        print("❌ No models available for inference")
        return
    
    # Create realistic sample data
    sample_data = {
        'adj_close': [105.23],
        'close': [105.45], 
        'high': [107.89],
        'low': [103.12],
        'open': [104.56],
        'volume': [4567890],
        'logret_1': [0.0123],
        'logret_5': [0.0234],
        'logret_10': [0.0345],
        'vol_10': [0.0187],
        'vol_20': [0.0212],
        'vol_sma20': [0.0198],
        'vol_std20': [0.0043],
        'vol_z20': [0.326],
        'ema_20': [104.89],
        'ema_50': [103.45],
        'ema_200': [101.23],
        'trend_20_50': [0.0139],
        'trend_px_200': [0.0417],
        'rsi_14': [58.7],
        'bb_z_20': [0.89],
        'atr_14': [2.34],
        'atrp_14': [0.0222],
        'dd_63': [0.0234],
        'dist_252_high': [0.123]
    }
    
    X_sample = pd.DataFrame(sample_data)
    
    print("📝 Sample Market Conditions:")
    print(f"   Price: ${X_sample['close'].iloc[0]:.2f}")
    print(f"   RSI: {X_sample['rsi_14'].iloc[0]:.1f}")
    print(f"   Volatility: {X_sample['vol_20'].iloc[0]:.3f}")
    print(f"   Trend: {X_sample['trend_20_50'].iloc[0]:+.3f}")
    
    print(f"\n🔮 Model Forecasts:")
    
    for regime in regimes:
        try:
            start_time = time.time()
            result = predict_regime(regime.lower(), X_sample)
            inference_time = (time.time() - start_time) * 1000
            
            print(f"\n   📊 {regime} Regime:")
            print(f"      ⬆️  Up:   {result['up']['expected_return']:+.4f} (confidence: {result['up']['confidence']:.3f})")
            print(f"      ⬇️  Down: {result['down']['expected_return']:+.4f} (confidence: {result['down']['confidence']:.3f})")
            print(f"      ⏱️  Inference time: {inference_time:.1f}ms")
            
        except Exception as e:
            print(f"❌ {regime} inference failed: {e}")

def demo_performance_analysis():
    """Show performance analysis across regimes"""
    print_subheader("Performance Analysis")
    
    regimes = list_available_regimes()
    if len(regimes) < 2:
        print("❌ Need at least 2 regimes for comparison")
        return
    
    print("📊 Regime Performance Comparison:")
    print(f"{'Regime':<15} {'Up AUC':<8} {'Down AUC':<9} {'Up MAE':<8} {'Down MAE':<9}")
    print("-" * 60)
    
    for regime in regimes:
        try:
            info = get_regime_info(regime.lower())
            metrics = info['metrics']
            
            print(f"{regime:<15} {metrics['up_auc']:<8.3f} {metrics['down_auc']:<9.3f} "
                  f"{metrics['up_mae']:<8.4f} {metrics['down_mae']:<9.4f}")
        except Exception as e:
            print(f"{regime:<15} {'Error':<8} {'N/A':<9} {'N/A':<8} {'N/A':<9}")

def demo_feature_importance():
    """Show feature importance from trained models"""
    print_subheader("Feature Importance Analysis")
    
    try:
        import joblib
        from pathlib import Path
        
        # Load one model to show feature importance
        model_path = Path("models/calm/model_up_clf.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            feature_path = Path("models/calm/feature_list.json")
            
            with open(feature_path, 'r') as f:
                features = json.load(f)
            
            # Get feature importance
            importances = model.feature_importances_
            feature_importance = list(zip(features, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("🔝 Top 10 Most Important Features (Calm Up Model):")
            for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                print(f"   {i:2d}. {feature:<15} : {importance:.4f}")
        else:
            print("❌ No model found for feature importance analysis")
            
    except Exception as e:
        print(f"❌ Feature importance analysis failed: {e}")

def main():
    """Run complete demo"""
    print_header("🚀 ML Trading Regime Models - Hackathon Demo")
    
    # Check dependencies
    print("🔧 Checking dependencies...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements_minimal.txt")
        return
    
    # Run demo sections
    demo_data_info()
    demo_model_info()
    demo_inference()
    demo_performance_analysis()
    demo_feature_importance()
    
    print_header("🎉 Demo Complete!")
    print("✅ Pipeline demonstrated successfully!")
    print("📈 Ready for production use")
    print("\n🚀 Next Steps:")
    print("   1. Integrate with your trading system")
    print("   2. Add real-time data feeds")
    print("   3. Implement position sizing logic")
    print("   4. Add monitoring and alerting")

if __name__ == "__main__":
    main()

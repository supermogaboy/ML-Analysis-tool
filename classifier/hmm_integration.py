import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent))

from Features import create_features
from HMMRegimeClassifier import HMMRegimeClassifier
from Regimes import add_regime_features


def integrate_hmm_with_pipeline(df, model_path=None, n_regimes=4, train_new=True):
    """
    Integrate HMM regime classifier with existing data pipeline
    
    Args:
        df: DataFrame with OHLCV data
        model_path: Path to save/load HMM model
        n_regimes: Number of regimes for HMM
        train_new: Whether to train new model or load existing
        
    Returns:
        DataFrame with HMM regimes added
    """
    print("Integrating HMM Regime Classifier with pipeline...")
    
    # Initialize HMM classifier
    hmm_classifier = HMMRegimeClassifier(n_regimes=n_regimes)
    
    if train_new:
        # Train new HMM model
        print("Training new HMM regime classifier...")
        hmm_classifier.fit(df)
        
        # Save model if path provided
        if model_path:
            hmm_classifier.save(model_path)
    else:
        # Load existing model
        if not model_path or not Path(model_path).exists():
            raise ValueError("Model path required when train_new=False")
        hmm_classifier.load(model_path)
    
    # Predict regimes
    print("Predicting regimes with HMM...")
    df['hmm_regime'] = hmm_classifier.predict(df)
    
    # Get regime probabilities
    regime_probs = hmm_classifier.predict_proba(df)
    for col in regime_probs.columns:
        df.loc[:, f'hmm_prob_{col}'] = regime_probs[col]
    
    # Add regime-specific features
    df = add_regime_features(df)
    
    # Analyze regime stability
    stability = hmm_classifier.analyze_regime_stability(df)
    
    print(f"\nHMM Regime Analysis:")
    print(f"Regime Distribution: {stability['regime_distribution']}")
    print(f"Transition Matrix:\n{stability['transition_matrix']}")
    
    return df, hmm_classifier, stability


def compare_regime_methods(df, hmm_model_path="models/hmm_regime_classifier.pkl"):
    """
    Compare HMM regimes with existing regime detection methods
    
    Args:
        df: DataFrame with OHLCV data
        hmm_model_path: Path for HMM model
        
    Returns:
        DataFrame with multiple regime classifications
    """
    print("Comparing regime detection methods...")
    
    # Get existing regimes (from Regimes.py)
    from Regimes import detect_regimes
    df_existing = detect_regimes(df.copy(), use_classifier=True)
    
    # Get HMM regimes
    df_hmm, hmm_classifier, _ = integrate_hmm_with_pipeline(
        df.copy(), 
        model_path=hmm_model_path,
        train_new=True
    )
    
    # Combine results
    df_comparison = df.copy()
    df_comparison['existing_regime'] = df_existing['regime']
    df_comparison['hmm_regime'] = df_hmm['hmm_regime']
    
    # Add HMM probabilities
    for col in df_hmm.columns:
        if col.startswith('hmm_prob_'):
            df_comparison.loc[:, col] = df_hmm[col]
    
    # Create comparison analysis
    print("\nRegime Comparison Analysis:")
    print("Existing Regime Distribution:")
    print(df_comparison['existing_regime'].value_counts(normalize=True))
    
    print("\nHMM Regime Distribution:")
    print(df_comparison['hmm_regime'].value_counts(normalize=True))
    
    # Cross-tabulation
    print("\nCross-tabulation (Existing vs HMM):")
    cross_tab = pd.crosstab(
        df_comparison['existing_regime'], 
        df_comparison['hmm_regime'], 
        normalize='index'
    )
    print(cross_tab.round(3))
    
    return df_comparison, hmm_classifier


def create_enhanced_features_with_hmm(df, hmm_model_path="models/hmm_regime_classifier.pkl"):
    """
    Create enhanced features using HMM regime detection
    
    Args:
        df: DataFrame with OHLCV data
        hmm_model_path: Path for HMM model
        
    Returns:
        DataFrame with enhanced features
    """
    print("Creating enhanced features with HMM regime detection...")
    
    # Get basic features
    df = create_features(df)
    
    # Add HMM regimes
    df, hmm_classifier, stability = integrate_hmm_with_pipeline(
        df, 
        model_path=hmm_model_path,
        train_new=True
    )
    
    # Create regime-specific features
    df.loc[:, 'regime_vol_ratio'] = df.groupby('hmm_regime')['vol_20'].transform(
        lambda x: x / x.mean()
    )
    
    df.loc[:, 'regime_trend_strength'] = df.groupby('hmm_regime')['trend_20_50'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Create regime transition indicators
    df.loc[:, 'regime_changed'] = df['hmm_regime'] != df['hmm_regime'].shift(1)
    df.loc[:, 'regime_change_strength'] = 0.0
    
    for i in range(1, len(df)):
        if df['regime_changed'].iloc[i]:
            # Get probability change
            prev_probs = [col for col in df.columns if col.startswith('hmm_prob_')][0]
            curr_probs = [col for col in df.columns if col.startswith('hmm_prob_')][1:]
            
            if len(curr_probs) > 0:
                prob_change = abs(df[curr_probs[0]].iloc[i] - df[curr_probs[0]].iloc[i-1])
                df.loc[df.index[i], 'regime_change_strength'] = prob_change
    
    # Add regime persistence features
    df.loc[:, 'regime_persistence'] = df.groupby('hmm_regime').cumcount() + 1
    
    print(f"Enhanced features created. Total columns: {len(df.columns)}")
    
    return df, hmm_classifier


def backtest_hmm_regime_strategy(df, hmm_model_path="models/hmm_regime_classifier.pkl"):
    """
    Simple backtest of regime-based strategy using HMM
    
    Args:
        df: DataFrame with OHLCV data
        hmm_model_path: Path for HMM model
        
    Returns:
        DataFrame with backtest results
    """
    print("Running HMM regime strategy backtest...")
    
    # Get HMM regimes and features
    df, hmm_classifier, _ = integrate_hmm_with_pipeline(
        df.copy(), 
        model_path=hmm_model_path,
        train_new=True
    )
    
    # Simple regime-based strategy
    df.loc[:, 'strategy_return'] = 0.0
    
    # Define strategy logic per regime
    for regime in df['hmm_regime'].unique():
        if pd.isna(regime):
            continue
            
        regime_mask = df['hmm_regime'] == regime
        
        if regime == "Calm":
            # In calm markets, use mean reversion
            df.loc[regime_mask, 'strategy_return'] = -df.loc[regime_mask, 'logret_1'].shift(1)
            
        elif regime == "Trending":
            # In trending markets, follow the trend
            df.loc[regime_mask, 'strategy_return'] = np.where(
                df.loc[regime_mask, 'trend_20_50'] > 0,
                df.loc[regime_mask, 'logret_1'],
                -df.loc[regime_mask, 'logret_1']
            )
            
        elif regime == "Volatile_Event":
            # In volatile periods, stay in cash (zero return)
            df.loc[regime_mask, 'strategy_return'] = 0.0
            
        elif regime == "Crisis":
            # In crisis, defensive positioning (short bias)
            df.loc[regime_mask, 'strategy_return'] = -0.5 * df.loc[regime_mask, 'logret_1']
    
    # Calculate cumulative returns
    df.loc[:, 'strategy_cumret'] = np.exp(df['strategy_return'].cumsum())
    df.loc[:, 'buy_hold_cumret'] = np.exp(df['logret_1'].cumsum())
    
    # Calculate performance metrics
    strategy_ret = df['strategy_return'].mean() * 252  # Annualized
    strategy_vol = df['strategy_return'].std() * np.sqrt(252)
    strategy_sharpe = strategy_ret / strategy_vol if strategy_vol > 0 else 0
    
    buy_hold_ret = df['logret_1'].mean() * 252
    buy_hold_vol = df['logret_1'].std() * np.sqrt(252)
    buy_hold_sharpe = buy_hold_ret / buy_hold_vol if buy_hold_vol > 0 else 0
    
    print(f"\nStrategy Performance:")
    print(f"Annual Return: {strategy_ret:.2%}")
    print(f"Annual Volatility: {strategy_vol:.2%}")
    print(f"Sharpe Ratio: {strategy_sharpe:.2f}")
    
    print(f"\nBuy & Hold Performance:")
    print(f"Annual Return: {buy_hold_ret:.2%}")
    print(f"Annual Volatility: {buy_hold_vol:.2%}")
    print(f"Sharpe Ratio: {buy_hold_sharpe:.2f}")
    
    return df


if __name__ == "__main__":
    # Example usage with your data
    print("HMM Regime Classifier Integration")
    print("=" * 50)
    
    try:
        # Load your existing data
        df = pd.read_csv("qqq_features_with_regime.csv", parse_dates=["date"])
        df = df.set_index("date")
        
        # Ensure we have OHLCV data
        if 'close' not in df.columns:
            print("Error: 'close' column not found in data")
            exit(1)
        
        print(f"Loaded data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        
        # Option 1: Basic integration
        print("\n1. Basic HMM Integration:")
        df_basic, hmm_classifier, stability = integrate_hmm_with_pipeline(
            df.head(1000),  # Use subset for testing
            model_path="models/hmm_regime_classifier.pkl",
            train_new=True
        )
        
        # Option 2: Compare with existing methods
        print("\n2. Comparing with existing regime detection:")
        df_comparison, _ = compare_regime_methods(
            df.head(1000),
            hmm_model_path="models/hmm_regime_classifier.pkl"
        )
        
        # Option 3: Enhanced features
        print("\n3. Creating enhanced features:")
        df_enhanced, _ = create_enhanced_features_with_hmm(
            df.head(1000),
            hmm_model_path="models/hmm_regime_classifier.pkl"
        )
        
        # Option 4: Simple backtest
        print("\n4. Running strategy backtest:")
        df_backtest = backtest_hmm_regime_strategy(
            df.head(1000),
            hmm_model_path="models/hmm_regime_classifier.pkl"
        )
        
        print("\nIntegration completed successfully!")
        
    except FileNotFoundError:
        print("Error: qqq_features_with_regime.csv not found")
        print("Please ensure the data file exists in the current directory")
    except Exception as e:
        print(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from Features import create_features
from Regimes import add_regime_features


class HMMRegimeClassifier:
    """
    Enhanced Hidden Markov Model regime classifier that combines HMM with
    additional features for better regime detection.
    
    Regimes identified:
    - 0: Calm/Low volatility
    - 1: Normal/Trending
    - 2: Volatile Event
    - 3: Crisis/High volatility
    """
    
    def __init__(self, n_regimes=4, random_state=42, covariance_type="full"):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.regime_labels = {}
        self.feature_columns = None
        
    def _prepare_features(self, df):
        """
        Prepare features for HMM training using existing feature engineering
        """
        # Use existing feature engineering pipeline
        df_features = create_features(df.copy())
        
        # Select most relevant features for regime detection
        feature_cols = [
            'logret_1', 'vol_10', 'vol_20', 'vol_z20',
            'trend_20_50', 'trend_px_200', 'rsi_14', 
            'bb_z_20', 'atrp_14', 'dd_63'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        if len(available_features) < 3:
            raise ValueError("Insufficient features for HMM training")
            
        self.feature_columns = available_features
        return df_features[available_features].dropna()
    
    def fit(self, df, price_col='close'):
        """
        Train the HMM regime classifier
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
        """
        print("Preparing features for HMM training...")
        
        # Prepare features
        features = self._prepare_features(df)
        
        if len(features) < 100:
            raise ValueError(f"Insufficient data for HMM training: {len(features)} samples")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        print(f"Training HMM with {self.n_regimes} regimes on {len(features)} samples...")
        
        # Initialize and train HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=200,
            random_state=self.random_state,
            verbose=False
        )
        
        self.hmm_model.fit(features_scaled)
        
        # Get regime assignments for training data
        regimes = self.hmm_model.predict(features_scaled)
        features['regime'] = regimes
        
        # Label regimes based on characteristics
        self._label_regimes(features)
        
        print(f"HMM training completed. Regime labels: {self.regime_labels}")
        
        return self
    
    def _label_regimes(self, features_with_regimes):
        """
        Label regimes based on their statistical characteristics
        """
        regime_stats = features_with_regimes.groupby('regime').agg({
            'vol_20': ['mean', 'std'],
            'logret_1': ['mean', 'std'],
            'trend_20_50': 'mean',
            'dd_63': 'mean'
        }).round(4)
        
        print("\nRegime Characteristics:")
        print(regime_stats)
        
        # Initialize regime labels
        self.regime_labels = {}
        
        # Sort regimes by volatility (primary characteristic)
        vol_means = regime_stats[('vol_20', 'mean')].sort_values()
        
        # Assign labels based on multiple criteria to ensure distinct regimes
        for i, regime in enumerate(vol_means.index):
            vol_mean = vol_means.loc[regime]
            trend_mean = regime_stats.loc[regime, ('trend_20_50', 'mean')]
            dd_mean = regime_stats.loc[regime, ('dd_63', 'mean')]
            
            # Multi-criteria labeling to get 4 distinct regimes
            if vol_mean < 0.01 and abs(trend_mean) < 0.02:
                self.regime_labels[regime] = "Calm"
            elif vol_mean > 0.025 or dd_mean < -0.15:
                self.regime_labels[regime] = "Crisis"
            elif vol_mean > 0.015 and abs(trend_mean) > 0.025:
                self.regime_labels[regime] = "Volatile_Event"
            else:
                self.regime_labels[regime] = "Trending"
        
        # Ensure we have distinct labels - if duplicates exist, force assignment
        used_labels = set(self.regime_labels.values())
        expected_labels = ["Calm", "Trending", "Volatile_Event", "Crisis"]
        
        if len(used_labels) < len(expected_labels):
            # Force assign missing labels based on volatility ranking
            sorted_regimes = vol_means.index.tolist()
            missing_labels = set(expected_labels) - used_labels
            
            for i, regime in enumerate(sorted_regimes):
                if len(missing_labels) == 0:
                    break
                if self.regime_labels[regime] in used_labels and list(self.regime_labels.values()).count(self.regime_labels[regime]) > 1:
                    # This is a duplicate, replace with missing label
                    self.regime_labels[regime] = missing_labels.pop()
                    used_labels = set(self.regime_labels.values())
    
    def predict(self, df):
        """
        Predict regimes for new data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with regime predictions
        """
        if self.hmm_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare features
        features = self._prepare_features(df)
        
        if len(features) == 0:
            return pd.Series(index=df.index, dtype=object)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict regimes
        regimes = self.hmm_model.predict(features_scaled)
        
        # Map to regime labels
        regime_names = pd.Series(regimes, index=features.index).map(self.regime_labels)
        
        # Create full series with original index
        result = pd.Series(index=df.index, dtype=object)
        result.loc[regime_names.index] = regime_names
        
        return result
    
    def predict_proba(self, df):
        """
        Predict regime probabilities
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime probabilities
        """
        if self.hmm_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare features
        features = self._prepare_features(df)
        
        if len(features) == 0:
            return pd.DataFrame(index=df.index)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities
        probs = self.hmm_model.predict_proba(features_scaled)
        
        # Create DataFrame with regime names as columns
        prob_df = pd.DataFrame(
            probs,
            index=features.index,
            columns=[f"Regime_{i}" for i in range(self.n_regimes)]
        )
        
        # Map to regime labels, handling duplicates
        label_mapping = {}
        used_labels = set()
        
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"Regime_{i}")
            
            # Handle duplicate labels by adding suffix
            if label in used_labels:
                suffix = 1
                while f"{label}_{suffix}" in used_labels:
                    suffix += 1
                label = f"{label}_{suffix}"
            
            used_labels.add(label)
            label_mapping[f"Regime_{i}"] = label
        
        prob_df = prob_df.rename(columns=label_mapping)
        
        # Create full DataFrame with original index
        result = pd.DataFrame(index=df.index, columns=prob_df.columns)
        result.loc[prob_df.index] = prob_df
        
        return result
    
    def get_transition_matrix(self):
        """
        Get the regime transition matrix
        
        Returns:
            DataFrame with transition probabilities
        """
        if self.hmm_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Create labeled transition matrix
        transmat = pd.DataFrame(
            self.hmm_model.transmat_,
            index=[self.regime_labels.get(i, f"Regime_{i}") for i in range(self.n_regimes)],
            columns=[self.regime_labels.get(i, f"Regime_{i}") for i in range(self.n_regimes)]
        )
        
        return transmat
    
    def analyze_regime_stability(self, df):
        """
        Analyze regime stability and persistence
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with stability metrics
        """
        regimes = self.predict(df)
        
        # Calculate regime durations
        regime_durations = {}
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
                
            mask = regimes == regime
            changes = mask.astype(int).diff().fillna(0)
            starts = changes[changes == 1].index
            ends = changes[changes == -1].index
            
            if len(starts) > 0:
                if len(ends) == len(starts):
                    durations = [(end - start).days for start, end in zip(starts, ends)]
                else:
                    # Handle ongoing regime at the end
                    durations = [(end - start).days for start, end in zip(starts, ends[:-1])]
                    durations.append((regimes.index[-1] - starts[-1]).days)
                
                regime_durations[regime] = {
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'num_periods': len(durations)
                }
        
        return {
            'regime_durations': regime_durations,
            'transition_matrix': self.get_transition_matrix(),
            'regime_distribution': regimes.value_counts(normalize=True).to_dict()
        }
    
    def plot_regimes(self, df, price_col='close', figsize=(15, 10)):
        """
        Plot price data with regime classifications
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            figsize: Figure size
        """
        regimes = self.predict(df)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Price with regime backgrounds
        ax1 = axes[0]
        ax1.plot(df.index, df[price_col], 'k-', linewidth=1, label='Price')
        
        # Add regime backgrounds
        colors = {'Calm': 'lightblue', 'Trending': 'lightgreen', 
                 'Volatile_Event': 'orange', 'Crisis': 'lightcoral'}
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            mask = regimes == regime
            ax1.fill_between(df.index, df[price_col].min(), df[price_col].max(), 
                           where=mask, alpha=0.3, color=colors.get(regime, 'gray'), 
                           label=regime)
        
        ax1.set_title('Price with Regime Classifications')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volatility with regimes
        ax2 = axes[1]
        if 'vol_20' in df.columns:
            ax2.plot(df.index, df['vol_20'], 'b-', linewidth=1, label='Volatility')
            ax2.set_title('20-day Volatility')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime probabilities
        ax3 = axes[2]
        probs = self.predict_proba(df)
        if not probs.empty:
            for col in probs.columns:
                ax3.plot(probs.index, probs[col], linewidth=2, label=col)
            ax3.set_title('Regime Probabilities')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if self.hmm_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'regime_labels': self.regime_labels,
            'feature_columns': self.feature_columns,
            'n_regimes': self.n_regimes,
            'random_state': self.random_state,
            'covariance_type': self.covariance_type
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        print(f"HMM Regime Classifier saved to {path}")
    
    def load(self, path):
        """
        Load a trained model
        
        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        
        self.hmm_model = model_data['hmm_model']
        self.scaler = model_data['scaler']
        self.regime_labels = model_data['regime_labels']
        self.feature_columns = model_data['feature_columns']
        self.n_regimes = model_data['n_regimes']
        self.random_state = model_data['random_state']
        self.covariance_type = model_data['covariance_type']
        
        print(f"HMM Regime Classifier loaded from {path}")


def train_hmm_classifier(df, model_path="models/hmm_regime_classifier.pkl", n_regimes=4):
    """
    Convenience function to train and save an HMM regime classifier
    
    Args:
        df: DataFrame with OHLCV data
        model_path: Path to save the trained model
        n_regimes: Number of regimes to identify
        
    Returns:
        Trained HMMRegimeClassifier instance
    """
    classifier = HMMRegimeClassifier(n_regimes=n_regimes)
    classifier.fit(df)
    
    if model_path:
        classifier.save(model_path)
    
    return classifier


if __name__ == "__main__":
    # Example usage
    print("HMM Regime Classifier - Example Usage")
    print("=" * 50)
    
    # Load sample data
    try:
        df = pd.read_csv("qqq_features_with_regime.csv", parse_dates=["date"])
        df = df.set_index("date")
        print(f"Loaded data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        
        # Train classifier
        classifier = train_hmm_classifier(df, n_regimes=4)
        
        # Analyze regimes
        stability = classifier.analyze_regime_stability(df)
        print("\nRegime Distribution:")
        for regime, prob in stability['regime_distribution'].items():
            print(f"  {regime}: {prob:.1%}")
        
        print("\nTransition Matrix:")
        print(classifier.get_transition_matrix())
        
    except FileNotFoundError:
        print("Sample data file not found. Please ensure qqq_features_with_regime.csv exists.")

"""
Machine Learning-Based Data Synthesis and Augmentation for UUO Dosing
Uses existing metadata to generate synthetic but realistic dosing protocols
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import minimize
import joblib
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MLDosingDataGenerator:
    """Machine Learning-based dosing data generator with probabilistic validation"""
    
    def __init__(self, initial_data_path: str):
        self.data_path = Path(initial_data_path)
        self.output_dir = self.data_path.parent
        self.models = {}
        self.scalers = {}
        self.validation_threshold = 0.7  # Probabilistic expectation threshold
        
    def load_initial_data(self) -> pd.DataFrame:
        """Load initial literature database"""
        df = pd.read_csv(self.data_path)
        print(f"📥 Loaded {len(df)} initial records")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create engineered features for ML training"""
        
        df_feat = df.copy()
        
        # Dosing features
        df_feat['log_dose'] = np.log1p(df_feat['dose_mg_kg'])
        df_feat['total_exposure'] = df_feat['dose_mg_kg'] * df_feat['total_doses']
        df_feat['log_total_exposure'] = np.log1p(df_feat['total_exposure'])
        df_feat['dose_per_day'] = df_feat['dose_mg_kg'] * df_feat['total_doses'] / df_feat['duration_days']
        
        # Timing features
        df_feat['early_start'] = (df_feat['start_day'] == 0).astype(int)
        df_feat['treatment_density'] = df_feat['total_doses'] / df_feat['duration_days']
        
        # One-hot encode compound type
        compound_dummies = pd.get_dummies(df_feat['compound_type'], prefix='type')
        df_feat = pd.concat([df_feat, compound_dummies], axis=1)
        
        # Define feature columns
        feature_cols = [
            'log_dose', 'total_exposure', 'log_total_exposure', 'dose_per_day',
            'duration_days', 'early_start', 'treatment_density'
        ] + list(compound_dummies.columns)
        
        return df_feat, feature_cols
    
    def train_outcome_models(self, df: pd.DataFrame, feature_cols: List[str]):
        """Train ML models to predict outcomes from dosing parameters"""
        
        print("\n🤖 Training ML models for outcome prediction...")
        
        outcomes = ['creatinine_change_pct', 'bun_change_pct', 'fibrosis_score', 
                   'inflammation_score', 'efficacy_score', 'safety_score']
        
        X = df[feature_cols].values
        
        for outcome in outcomes:
            y = df[outcome].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[outcome] = scaler
            
            # Train Gradient Boosting model (best for small datasets)
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=2,
                random_state=42
            )
            model.fit(X_scaled, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), 
                                       scoring='r2')
            
            self.models[outcome] = model
            
            print(f"  ✓ {outcome}: R² = {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Save models
        model_path = self.output_dir / "ml_models"
        model_path.mkdir(exist_ok=True)
        
        for outcome, model in self.models.items():
            joblib.dump(model, model_path / f"{outcome}_model.pkl")
            joblib.dump(self.scalers[outcome], model_path / f"{outcome}_scaler.pkl")
        
        print(f"\n💾 Models saved to {model_path}")
    
    def generate_synthetic_candidates(self, n_candidates: int = 50) -> pd.DataFrame:
        """Generate synthetic dosing protocols using domain knowledge"""
        
        print(f"\n🧬 Generating {n_candidates} synthetic dosing candidates...")
        
        synthetic_data = []
        
        # Define realistic parameter ranges based on literature
        dose_ranges = {
            "JAK_inhibitor": (10, 60),
            "ARB": (1, 20),
            "antidiabetic": (50, 300),
            "SGLT2_inhibitor": (3, 30),
            "renin_inhibitor": (25, 100),
            "MR_blocker": (50, 200),
            "antioxidant": (50, 200),
            "antifibrotic": (100, 800),
            "senolytic": (25, 100),
        }
        
        compound_types = list(dose_ranges.keys())
        
        for i in range(n_candidates):
            # Randomly select compound type
            compound_type = np.random.choice(compound_types)
            dose_range = dose_ranges[compound_type]
            
            # Generate dosing parameters
            dose_mg_kg = np.random.uniform(dose_range[0], dose_range[1])
            duration_days = np.random.choice([7, 14, 21, 28])
            frequency = np.random.choice(["daily", "BID"], p=[0.8, 0.2])
            start_day = np.random.choice([0, -1, 1, 7], p=[0.6, 0.2, 0.1, 0.1])  # mostly concurrent
            
            freq_multiplier = 2 if frequency == "BID" else 1
            total_doses = duration_days * freq_multiplier
            
            candidate = {
                "compound_name": f"Synthetic_{compound_type}_{i+1}",
                "compound_type": compound_type,
                "dose_mg_kg": dose_mg_kg,
                "duration_days": duration_days,
                "frequency": frequency,
                "start_day": start_day,
                "total_doses": total_doses,
                "route": "oral",
                "dose_unit": "mg/kg",
                "species": "mouse",
                "strain": "C57BL/6",
                "sex": "male",
                "age_weeks": 8,
                "weight_grams": 25,
                "sample_size": 8,
                "data_source": "ml_generated",
                "confidence_level": "synthetic",
            }
            
            synthetic_data.append(candidate)
        
        df_synth = pd.DataFrame(synthetic_data)
        print(f"  ✓ Generated {len(df_synth)} candidates")
        
        return df_synth
    
    def predict_outcomes(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Predict outcomes for synthetic candidates using trained models"""
        
        print("\n🔮 Predicting outcomes for synthetic candidates...")
        
        # Engineer features
        df_feat, _ = self.engineer_features(df)
        
        # Ensure all required columns exist
        for col in feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0
        
        # Fill NaN values
        X = df_feat[feature_cols].fillna(0).values
        
        for outcome, model in self.models.items():
            # Scale features
            X_scaled = self.scalers[outcome].transform(X)
            
            # Predict
            predictions = model.predict(X_scaled)
            
            # Add realistic noise
            noise_std = max(predictions.std() * 0.15, 0.1)  # 15% noise with minimum
            predictions += np.random.normal(0, noise_std, size=len(predictions))
            
            df[outcome] = np.clip(predictions, 0, None)  # No negative values
        
        print("  ✓ Outcomes predicted for all candidates")
        
        return df
    
    def calculate_probability_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate probabilistic expectation score for each candidate"""
        
        print("\n🎲 Calculating probabilistic expectation scores...")
        
        # Fill NaN values first
        df = df.fillna(0)
        
        # Normalize scores to 0-1 range
        df['efficacy_norm'] = df['efficacy_score'] / 100
        df['safety_norm'] = df['safety_score'] / 100
        
        # Inverse normalize injury markers (lower is better)
        max_creat = df['creatinine_change_pct'].max() if df['creatinine_change_pct'].max() > 0 else 1
        max_bun = df['bun_change_pct'].max() if df['bun_change_pct'].max() > 0 else 1
        max_fib = df['fibrosis_score'].max() if df['fibrosis_score'].max() > 0 else 1
        max_infl = df['inflammation_score'].max() if df['inflammation_score'].max() > 0 else 1
        
        df['creat_norm'] = 1 - (df['creatinine_change_pct'] / max_creat)
        df['bun_norm'] = 1 - (df['bun_change_pct'] / max_bun)
        df['fib_norm'] = 1 - (df['fibrosis_score'] / max_fib)
        df['infl_norm'] = 1 - (df['inflammation_score'] / max_infl)
        
        # Weighted probability score (weights based on clinical relevance)
        weights = {
            'efficacy_norm': 0.30,
            'safety_norm': 0.20,
            'creat_norm': 0.15,
            'bun_norm': 0.10,
            'fib_norm': 0.15,
            'infl_norm': 0.10
        }
        
        df['probability_score'] = sum(df[col] * weight for col, weight in weights.items())
        
        # Calculate confidence interval
        n_outcomes = 6
        df['score_std'] = df[['efficacy_norm', 'safety_norm', 'creat_norm', 
                               'bun_norm', 'fib_norm', 'infl_norm']].std(axis=1)
        df['confidence_interval'] = 1.96 * df['score_std'] / np.sqrt(n_outcomes)
        
        print(f"  ✓ Mean probability score: {df['probability_score'].mean():.3f}")
        print(f"  ✓ Score range: [{df['probability_score'].min():.3f}, {df['probability_score'].max():.3f}]")
        
        return df
    
    def filter_by_threshold(self, df: pd.DataFrame, threshold: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter candidates by probabilistic threshold"""
        
        if threshold is None:
            threshold = self.validation_threshold
        
        print(f"\n🔍 Filtering candidates with probability score > {threshold}...")
        
        df_accepted = df[df['probability_score'] >= threshold].copy()
        df_rejected = df[df['probability_score'] < threshold].copy()
        
        print(f"  ✓ Accepted: {len(df_accepted)} candidates")
        print(f"  ✓ Rejected: {len(df_rejected)} candidates (will be re-analyzed)")
        
        return df_accepted, df_rejected
    
    def re_analyze_rejected(self, df_rejected: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Re-analyze rejected candidates with parameter optimization"""
        
        if len(df_rejected) == 0:
            return pd.DataFrame()
        
        print(f"\n🔄 Re-analyzing {len(df_rejected)} rejected candidates...")
        
        improved_candidates = []
        
        for idx, row in df_rejected.iterrows():
            # Optimize dosing parameters to improve probability score
            optimized = self._optimize_dosing_parameters(row, feature_cols)
            
            if optimized['probability_score'] >= self.validation_threshold:
                improved_candidates.append(optimized)
        
        df_improved = pd.DataFrame(improved_candidates)
        print(f"  ✓ Successfully improved {len(df_improved)} candidates")
        
        return df_improved
    
    def _optimize_dosing_parameters(self, params: pd.Series, feature_cols: List[str]) -> Dict:
        """Optimize dosing parameters to maximize probability score"""
        
        compound_type = params['compound_type']
        
        # Define bounds
        dose_ranges = {
            "JAK_inhibitor": (10, 60),
            "ARB": (1, 20),
            "antidiabetic": (50, 300),
            "SGLT2_inhibitor": (3, 30),
            "renin_inhibitor": (25, 100),
            "MR_blocker": (50, 200),
            "antioxidant": (50, 200),
            "antifibrotic": (100, 800),
            "senolytic": (25, 100),
        }
        
        dose_range = dose_ranges.get(compound_type, (10, 200))
        
        # Simple grid search for optimization
        best_score = 0
        best_params = params.to_dict()
        
        for dose in np.linspace(dose_range[0], dose_range[1], 5):
            for duration in [7, 14, 21]:
                # Create candidate
                candidate = params.copy()
                candidate['dose_mg_kg'] = dose
                candidate['duration_days'] = duration
                candidate['total_doses'] = duration  # assume daily
                
                # Predict outcomes
                df_temp = pd.DataFrame([candidate])
                df_temp = self.predict_outcomes(df_temp, feature_cols)
                df_temp = self.calculate_probability_score(df_temp)
                
                score = df_temp['probability_score'].values[0]
                
                if score > best_score:
                    best_score = score
                    best_params = df_temp.iloc[0].to_dict()
        
        return best_params


def main():
    """Main execution"""
    print("="*80)
    print("🤖 ML-Based Dosing Data Generation Pipeline")
    print("="*80)
    
    # Initialize generator
    data_path = "drug_discovery/literature_data/uuo_initial_database.csv"
    generator = MLDosingDataGenerator(data_path)
    
    # Load initial data
    df_initial = generator.load_initial_data()
    
    # Engineer features and train models
    df_feat, feature_cols = generator.engineer_features(df_initial)
    generator.train_outcome_models(df_feat, feature_cols)
    
    # Generate synthetic candidates
    df_synthetic = generator.generate_synthetic_candidates(n_candidates=50)
    
    # Predict outcomes
    df_synthetic = generator.predict_outcomes(df_synthetic, feature_cols)
    
    # Calculate probability scores
    df_synthetic = generator.calculate_probability_score(df_synthetic)
    
    # Filter by threshold
    df_accepted, df_rejected = generator.filter_by_threshold(df_synthetic)
    
    # Re-analyze rejected candidates
    df_improved = generator.re_analyze_rejected(df_rejected, feature_cols)
    
    # Combine all accepted data
    df_final = pd.concat([df_initial, df_accepted, df_improved], ignore_index=True)
    
    # Save final dataset
    output_path = generator.output_dir / "uuo_ml_generated_database.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ Final Dataset Summary:")
    print(f"  - Initial records: {len(df_initial)}")
    print(f"  - Synthetic accepted: {len(df_accepted)}")
    print(f"  - Improved from rejected: {len(df_improved)}")
    print(f"  - Total final records: {len(df_final)}")
    print(f"  - Mean probability score: {df_final['probability_score'].mean():.3f}")
    
    print(f"\n💾 Final dataset saved to: {output_path}")
    
    return df_final


if __name__ == "__main__":
    df = main()

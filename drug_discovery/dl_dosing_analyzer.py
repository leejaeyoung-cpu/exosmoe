"""
Deep Learning-Based Dosing Data Generation and Comparative Analysis
Uses advanced neural networks to generate and validate dosing protocols
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json


class UUODosingDataset(Dataset):
    """PyTorch Dataset for UUO dosing data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepDosingPredictor(nn.Module):
    """Deep neural network for dosing-outcome prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(DeepDosingPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (6 outcomes)
        layers.append(nn.Linear(prev_dim, 6))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class VariationalAutoencoder(nn.Module):
    """VAE for generating realistic dosing protocols"""
    
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DLDosingAnalyzer:
    """Deep Learning-based dosing analysis and generation"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.output_dir = self.data_path.parent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        self.predictor = None
        self.vae = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for deep learning"""
        
        df = pd.read_csv(self.data_path)
        print(f"📥 Loaded {len(df)} records")
        
        # Feature engineering
        df['log_dose'] = np.log1p(df['dose_mg_kg'])
        df['total_exposure'] = df['dose_mg_kg'] * df['total_doses']
        df['log_total_exposure'] = np.log1p(df['total_exposure'])
        df['dose_per_day'] = df['dose_mg_kg'] * df['total_doses'] / df['duration_days']
        df['early_start'] = (df['start_day'] <= 0).astype(int)
        df['treatment_density'] = df['total_doses'] / df['duration_days']
        
        # One-hot encode compound type
        compound_dummies = pd.get_dummies(df['compound_type'], prefix='type')
        df = pd.concat([df, compound_dummies], axis=1)
        
        # Select features
        feature_cols = [
            'log_dose', 'total_exposure', 'log_total_exposure', 'dose_per_day',
            'duration_days', 'early_start', 'treatment_density'
        ] + list(compound_dummies.columns)
        
        outcome_cols = ['creatinine_change_pct', 'bun_change_pct', 'fibrosis_score',
                       'inflammation_score', 'efficacy_score', 'safety_score']
        
        X = df[feature_cols].values
        y = df[outcome_cols].values
        
        return df, X, y, feature_cols
    
    def train_predictor(self, X: np.ndarray, y: np.ndarray, epochs: int = 200):
        """Train deep neural network predictor"""
        
        print("\n🧠 Training Deep Learning Predictor...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        # Create datasets
        train_dataset = UUODosingDataset(X_train_scaled, y_train_scaled)
        val_dataset = UUODosingDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # Initialize model
        self.predictor = DeepDosingPredictor(input_dim=X.shape[1]).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.predictor.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.predictor.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.predictor(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.predictor.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.predictor(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.predictor.state_dict(), 
                          self.output_dir / 'dl_predictor_best.pth')
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def train_vae(self, X: np.ndarray, epochs: int = 300):
        """Train VAE for data generation"""
        
        print("\n🎨 Training Variational Autoencoder...")
        
        X_scaled = self.scaler_X.transform(X)
        dataset = UUODosingDataset(X_scaled, X_scaled)  # Autoencoder: input = output
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Initialize VAE
        self.vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=16).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            self.vae.train()
            total_loss = 0
            
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = self.vae(X_batch)
                
                # VAE loss = reconstruction + KL divergence
                recon_loss = nn.functional.mse_loss(recon_batch, X_batch, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        
        torch.save(self.vae.state_dict(), self.output_dir / 'vae_generator.pth')
        print("✅ VAE training complete!")
    
    def generate_dl_candidates(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate new dosing candidates using VAE"""
        
        print(f"\n🎲 Generating {n_samples} candidates using Deep Learning...")
        
        self.vae.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, 16).to(self.device)
            generated = self.vae.decode(z).cpu().numpy()
        
        # Inverse transform
        generated_scaled = self.scaler_X.inverse_transform(generated)
        
        print(f"✅ Generated {n_samples} candidates")
        
        return generated_scaled
    
    def predict_outcomes_dl(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes using deep learning model"""
        
        self.predictor.eval()
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = self.predictor(X_tensor).cpu().numpy()
        
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def _plot_training_history(self, train_losses: List[float], val_losses: List[float]):
        """Plot training history"""
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Deep Learning Model Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / 'dl_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved to {save_path}")
        plt.close()


def compare_with_literature(df_ml: pd.DataFrame, df_dl: pd.DataFrame, 
                           output_dir: Path) -> Dict:
    """Compare ML and DL generated data with literature"""
    
    print("\n📊 Comparing ML vs DL Generated Data...")
    
    # Statistical comparison
    comparison = {
        "ml_stats": {
            "n_samples": len(df_ml),
            "mean_efficacy": float(df_ml['efficacy_score'].mean()),
            "mean_safety": float(df_ml['safety_score'].mean()),
            "mean_prob_score": float(df_ml['probability_score'].mean()),
        },
        "dl_stats": {
            "n_samples": len(df_dl),
            "mean_efficacy": float(df_dl['efficacy_score'].mean()),
            "mean_safety": float(df_dl['safety_score'].mean()),
        },
        "significant_differences": []
    }
    
    # Identify significant differences
    efficacy_diff = abs(comparison['ml_stats']['mean_efficacy'] - 
                       comparison['dl_stats']['mean_efficacy'])
    
    if efficacy_diff > 5:
        comparison['significant_differences'].append({
            "metric": "efficacy_score",
            "difference": float(efficacy_diff),
            "interpretation": "Significant difference in predicted efficacy"
        })
    
    # Save comparison
    with open(output_dir / 'ml_vs_dl_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("✅ Comparison complete")
    
    return comparison


def main():
    """Main execution"""
    print("="*80)
    print("🧠 Deep Learning-Based Dosing Analysis Pipeline")
    print("="*80)
    
    # Initialize analyzer
    data_path = "drug_discovery/literature_data/uuo_ml_generated_database.csv"
    analyzer = DLDosingAnalyzer(data_path)
    
    # Load data
    df, X, y, feature_cols = analyzer.load_data()
    
    # Train predictor
    analyzer.train_predictor(X, y, epochs=200)
    
    # Train VAE
    analyzer.train_vae(X, epochs=300)
    
    # Generate new candidates
    X_generated = analyzer.generate_dl_candidates(n_samples=100)
    
    # Predict outcomes for generated data
    y_predicted = analyzer.predict_outcomes_dl(X_generated)
    
    # Create DataFrame for DL-generated data
    df_dl = pd.DataFrame(X_generated, columns=feature_cols)
    outcome_cols = ['creatinine_change_pct', 'bun_change_pct', 'fibrosis_score',
                   'inflammation_score', 'efficacy_score', 'safety_score']
    df_dl[outcome_cols] = y_predicted
    
    # Save DL-generated data
    output_path = analyzer.output_dir / 'uuo_dl_generated_database.csv'
    df_dl.to_csv(output_path, index=False)
    
    print(f"\n✅ DL Pipeline Complete!")
    print(f"  - Total DL-generated records: {len(df_dl)}")
    print(f"  - Mean predicted efficacy: {df_dl['efficacy_score'].mean():.1f}")
    print(f"  - Mean predicted safety: {df_dl['safety_score'].mean():.1f}")
    print(f"\n💾 Data saved to: {output_path}")
    
    return df_dl


if __name__ == "__main__":
    df_dl = main()

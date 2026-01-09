"""
NOVA Lead Validation - Deep Learning Predictor
SMILES → 실험 결과 예측 모델
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, Tuple

class MoleculeDataset(Dataset):
    """분자 데이터셋"""
    
    def __init__(self, smiles_list, targets, fingerprint_size=2048):
        self.smiles = smiles_list
        self.targets = targets
        self.fingerprint_size = fingerprint_size
        
        # Generate fingerprints
        self.fingerprints = []
        for smi in smiles_list:
            fp = self.smiles_to_fingerprint(smi)
            self.fingerprints.append(fp)
        
        self.fingerprints = np.array(self.fingerprints, dtype=np.float32)
        
    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        """SMILES → Morgan Fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.fingerprint_size)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fingerprint_size)
        return np.array(fp, dtype=np.float32)
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return {
            'fingerprint': torch.FloatTensor(self.fingerprints[idx]),
            'targets': torch.FloatTensor(self.targets[idx])
        }


class MultiTaskDNN(nn.Module):
    """
    Multi-Task Deep Neural Network
    
    Input: Molecular fingerprint (2048-bit)
    Output: Multiple experimental readouts
    """
    
    def __init__(self, input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=10):
        super(MultiTaskDNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'caga_ic50': nn.Linear(prev_size, 1),  # Regression
            'nfkb_ic50': nn.Linear(prev_size, 1),  # Regression
            'psmad_inhib': nn.Linear(prev_size, 1),  # Regression
            'pikba_inhib': nn.Linear(prev_size, 1),  # Regression
            'fib_gene_red': nn.Linear(prev_size, 1),  # Regression
            'inf_gene_red': nn.Linear(prev_size, 1),  # Regression
            'cc50': nn.Linear(prev_size, 1),  # Regression
            'solubility': nn.Linear(prev_size, 1),  # Regression
            'gate1_go': nn.Linear(prev_size, 1),  # Binary classification
        })
    
    def forward(self, x):
        shared_rep = self.shared_layers(x)
        
        outputs = {}
        for task_name, head in self.task_heads.items():
            if task_name == 'gate1_go':
                outputs[task_name] = torch.sigmoid(head(shared_rep))
            else:
                outputs[task_name] = head(shared_rep)
        
        return outputs


class NOVAPredictor:
    """NOVA 후보 물질 실험 결과 예측기"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.scaler_X = StandardScaler()
        self.scalers_y = {}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """데이터 전처리"""
        smiles_list = df['smiles'].tolist()
        
        # Target columns
        target_cols = {
            'caga_ic50': 'CAGA_IC50_uM',
            'nfkb_ic50': 'NF-kB_IC50_uM',
            'psmad_inhib': 'pSMAD_inhibition',
            'pikba_inhib': 'pIkBa_inhibition',
            'fib_gene_red': 'fibrosis_genes_avg_reduction',
            'inf_gene_red': 'inflammation_genes_avg_reduction',
            'cc50': 'CC50_uM',
            'solubility': 'solubility_uM',
            'gate1_go': 'Gate1_GO',
        }
        
        targets = []
        for idx, row in df.iterrows():
            target_vals = [row[col] for col in target_cols.values()]
            targets.append(target_vals)
        
        targets = np.array(targets, dtype=np.float32)
        
        return smiles_list, targets, target_cols
    
    def train(self, df: pd.DataFrame, epochs=100, batch_size=32, lr=0.001):
        """모델 학습"""
        smiles_list, targets, target_cols = self.prepare_data(df)
        
        # Train/Val split
        smiles_train, smiles_val, targets_train, targets_val = train_test_split(
            smiles_list, targets, test_size=0.2, random_state=42
        )
        
        # Normalize targets (regression tasks only, not binary)
        for i, (task_name, col_name) in enumerate(target_cols.items()):
            if task_name != 'gate1_go':
                scaler = StandardScaler()
                targets_train[:, i] = scaler.fit_transform(targets_train[:, i].reshape(-1, 1)).flatten()
                targets_val[:, i] = scaler.transform(targets_val[:, i].reshape(-1, 1)).flatten()
                self.scalers_y[task_name] = scaler
        
        # Datasets
        train_dataset = MoleculeDataset(smiles_train, targets_train)
        val_dataset = MoleculeDataset(smiles_val, targets_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Model
        self.model = MultiTaskDNN(input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=len(target_cols))
        self.model = self.model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                fps = batch['fingerprint'].to(self.device)
                targets_batch = batch['targets'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(fps)
                
                # Multi-task loss
                loss = 0
                task_names = list(target_cols.keys())
                for i, task_name in enumerate(task_names):
                    pred = outputs[task_name].squeeze()
                    true = targets_batch[:, i]
                    
                    if task_name == 'gate1_go':
                        # Binary classification (BCE)
                        loss += F.binary_cross_entropy(pred, true)
                    else:
                        # Regression (MSE)
                        loss += F.mse_loss(pred, true)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    fps = batch['fingerprint'].to(self.device)
                    targets_batch = batch['targets'].to(self.device)
                    
                    outputs = self.model(fps)
                    
                    loss = 0
                    for i, task_name in enumerate(task_names):
                        pred = outputs[task_name].squeeze()
                        true = targets_batch[:, i]
                        
                        if task_name == 'gate1_go':
                            loss += F.binary_cross_entropy(pred, true)
                        else:
                            loss += F.mse_loss(pred, true)
                    
                    val_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('models/nova_best_model.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return history
    
    def predict(self, smiles_list: list) -> pd.DataFrame:
        """예측"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        self.model.eval()
        
        dataset = MoleculeDataset(smiles_list, np.zeros((len(smiles_list), 9)))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = {task: [] for task in ['caga_ic50', 'nfkb_ic50', 'psmad_inhib', 
                                                   'pikba_inhib', 'fib_gene_red', 'inf_gene_red',
                                                   'cc50', 'solubility', 'gate1_go']}
        
        with torch.no_grad():
            for batch in loader:
                fps = batch['fingerprint'].to(self.device)
                outputs = self.model(fps)
                
                for task_name, preds in outputs.items():
                    preds_np = preds.cpu().numpy().flatten()
                    
                    # Denormalize (regression tasks)
                    if task_name != 'gate1_go' and task_name in self.scalers_y:
                        preds_np = self.scalers_y[task_name].inverse_transform(preds_np.reshape(-1, 1)).flatten()
                    
                    all_predictions[task_name].extend(preds_np)
        
        # Results DataFrame
        results = pd.DataFrame({
            'smiles': smiles_list,
            'pred_CAGA_IC50_uM': all_predictions['caga_ic50'],
            'pred_NF-kB_IC50_uM': all_predictions['nfkb_ic50'],
            'pred_pSMAD_inhibition': all_predictions['psmad_inhib'],
            'pred_pIkBa_inhibition': all_predictions['pikba_inhib'],
            'pred_fibrosis_reduction': all_predictions['fib_gene_red'],
            'pred_inflammation_reduction': all_predictions['inf_gene_red'],
            'pred_CC50_uM': all_predictions['cc50'],
            'pred_solubility_uM': all_predictions['solubility'],
            'pred_Gate1_GO_prob': all_predictions['gate1_go'],
        })
        
        # GO/NO-GO decision
        results['pred_Gate1_GO'] = results['pred_Gate1_GO_prob'] > 0.5
        
        return results
    
    def save_model(self, path):
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'scalers_y': self.scalers_y,
        }, path)
        print(f"✅ Model saved: {path}")
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = MultiTaskDNN()
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)
        
        self.scalers_y = checkpoint['scalers_y']
        print(f"✅ Model loaded: {path}")


def main():
    """메인 실행"""
    # Load synthetic data
    data_path = Path("generated_molecules/synthetic_experimental_data.csv")
    
    if not data_path.exists():
        print("❌ Synthetic data not found. Run nova_ml_data_generator.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} molecules with synthetic experimental data")
    
    # Train model
    print("\n🚀 Training Deep Learning Model...")
    predictor = NOVAPredictor()
    history = predictor.train(df, epochs=50, batch_size=16, lr=0.001)
    
    print("\n✅ Training complete!")
    
    # Test prediction on Top 10
    print("\n🔮 Testing predictions on top 10 molecules...")
    top10_smiles = df['smiles'].head(10).tolist()
    predictions = predictor.predict(top10_smiles)
    
    print(predictions[['smiles', 'pred_Gate1_GO_prob', 'pred_Gate1_GO']].head(10))
    
    print(f"\n📈 Predicted GO rate: {predictions['pred_Gate1_GO'].sum() / len(predictions) * 100:.1f}%")


if __name__ == "__main__":
    main()

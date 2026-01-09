"""
NOVA IND-Enabling Deep Learning Predictor
5 Gate 통과 여부 + IND Success 예측 모델
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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pathlib import Path
from typing import Dict, Tuple

class MoleculeDatasetIND(Dataset):
    """IND Gate 예측용 분자 데이터셋"""
    
    def __init__(self, smiles_list, targets_dict, fingerprint_size=2048):
        self.smiles = smiles_list
        self.targets_dict = targets_dict
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
        targets = {}
        for key, values in self.targets_dict.items():
            targets[key] = torch.FloatTensor([values[idx]])
        
        return {
            'fingerprint': torch.FloatTensor(self.fingerprints[idx]),
            'targets': targets
        }


class INDGatePredictor(nn.Module):
    """
    IND Gate 예측 Multi-Task DNN
    
    Input: Molecular fingerprint (2048-bit)
    Output: 5 Gates + IND Success + Continuous metrics
    """
    
    def __init__(self, input_size=2048, hidden_sizes=[1024, 512, 256, 128]):
        super(INDGatePredictor, self).__init__()
        
        # Shared layers
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
            # Gate A
            'gate_a': nn.Linear(prev_size, 1),  # Binary
            'egfr_selectivity': nn.Linear(prev_size, 1),  # Regression
            'herg_ic50': nn.Linear(prev_size, 1),  # Regression
            'solubility': nn.Linear(prev_size, 1),  # Regression
            
            # Gate B
            'gate_b': nn.Linear(prev_size, 1),  # Binary
            'overall_yield': nn.Linear(prev_size, 1),  # Regression
            'max_impurity': nn.Linear(prev_size, 1),  # Regression
            
            # Gate C
            'gate_c': nn.Linear(prev_size, 1),  # Binary
            'noael_rat': nn.Linear(prev_size, 1),  # Regression
            'qtc_prolongation': nn.Linear(prev_size, 1),  # Regression
            'genotox': nn.Linear(prev_size, 1),  # Binary (all negative)
            
            # Gate D
            'gate_d': nn.Linear(prev_size, 1),  # Binary
            'starting_dose': nn.Linear(prev_size, 1),  # Regression
            
            # Gate E
            'gate_e': nn.Linear(prev_size, 1),  # Binary
            'ind_completeness': nn.Linear(prev_size, 1),  # Regression
            
            # Overall
            'ind_success': nn.Linear(prev_size, 1),  # Binary (final)
            'ind_score': nn.Linear(prev_size, 1),  # Regression (0-100)
        })
    
    def forward(self, x):
        shared_rep = self.shared_layers(x)
        
        outputs = {}
        for task_name, head in self.task_heads.items():
            if task_name in ['gate_a', 'gate_b', 'gate_c', 'gate_d', 'gate_e', 'ind_success', 'genotox']:
                # Binary classification → sigmoid
                outputs[task_name] = torch.sigmoid(head(shared_rep))
            else:
                # Regression
                outputs[task_name] = head(shared_rep)
        
        return outputs


class NOVAINDPredictor:
    """NOVA IND Gate 예측기"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.scalers = {}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """데이터 전처리"""
        smiles_list = df['smiles'].tolist()
        
        # Target columns
        target_mapping = {
            # Binary gates
            'gate_a': 'Gate_A_PASS',
            'gate_b': 'Gate_B_PASS',
            'gate_c': 'Gate_C_PASS',
            'gate_d': 'Gate_D_PASS',
            'gate_e': 'Gate_E_PASS',
            'ind_success': 'IND_Success',
            'genotox': 'Genotox_All_Negative',
            
            # Continuous
            'egfr_selectivity': 'EGFR_Selectivity',
            'herg_ic50': 'hERG_IC50_uM',
            'solubility': 'Thermo_Solubility_uM',
            'overall_yield': 'Overall_Yield_Percent',
            'max_impurity': 'Max_Single_Impurity_Percent',
            'noael_rat': 'NOAEL_Rat_mg_kg',
            'qtc_prolongation': 'QTc_Prolongation_Percent',
            'starting_dose': 'Starting_Dose_mg',
            'ind_completeness': 'IND_Completeness_Percent',
            'ind_score': 'IND_Score',
        }
        
        targets_dict = {}
        for model_key, df_col in target_mapping.items():
            if df_col in df.columns:
                targets_dict[model_key] = df[df_col].values.astype(np.float32)
            else:
                targets_dict[model_key] = np.zeros(len(df), dtype=np.float32)
        
        return smiles_list, targets_dict, target_mapping
    
    def train(self, df: pd.DataFrame, epochs=100, batch_size=16, lr=0.001):
        """모델 학습"""
        smiles_list, targets_dict, target_mapping = self.prepare_data(df)
        
        # Train/Val split
        indices = np.arange(len(smiles_list))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        smiles_train = [smiles_list[i] for i in train_idx]
        smiles_val = [smiles_list[i] for i in val_idx]
        
        targets_train = {k: v[train_idx] for k, v in targets_dict.items()}
        targets_val = {k: v[val_idx] for k, v in targets_dict.items()}
        
        # Normalize continuous targets
        continuous_tasks = ['egfr_selectivity', 'herg_ic50', 'solubility', 'overall_yield',
                           'max_impurity', 'noael_rat', 'qtc_prolongation', 'starting_dose',
                           'ind_completeness', 'ind_score']
        
        for task in continuous_tasks:
            if task in targets_train:
                scaler = StandardScaler()
                targets_train[task] = scaler.fit_transform(targets_train[task].reshape(-1, 1)).flatten()
                targets_val[task] = scaler.transform(targets_val[task].reshape(-1, 1)).flatten()
                self.scalers[task] = scaler
        
        # Datasets
        train_dataset = MoleculeDatasetIND(smiles_train, targets_train)
        val_dataset = MoleculeDatasetIND(smiles_val, targets_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Model
        self.model = INDGatePredictor(input_size=2048, hidden_sizes=[1024, 512, 256, 128])
        self.model = self.model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                fps = batch['fingerprint'].to(self.device)
                targets_batch = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                optimizer.zero_grad()
                outputs = self.model(fps)
                
                # Multi-task loss
                loss = 0
                for task_name, pred in outputs.items():
                    if task_name in targets_batch:
                        true = targets_batch[task_name].squeeze()
                        pred = pred.squeeze()
                        
                        if task_name in ['gate_a', 'gate_b', 'gate_c', 'gate_d', 'gate_e', 'ind_success', 'genotox']:
                            # Binary → BCE
                            loss += F.binary_cross_entropy(pred, true)
                        else:
                            # Regression → MSE
                            loss += F.mse_loss(pred, true)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            val_preds = {'ind_success': [], 'gate_a': [], 'gate_c': []}
            val_trues = {'ind_success': [], 'gate_a': [], 'gate_c': []}
            
            with torch.no_grad():
                for batch in val_loader:
                    fps = batch['fingerprint'].to(self.device)
                    targets_batch = {k: v.to(self.device) for k, v in batch['targets'].items()}
                    
                    outputs = self.model(fps)
                    
                    loss = 0
                    for task_name, pred in outputs.items():
                        if task_name in targets_batch:
                            true = targets_batch[task_name].squeeze()
                            pred_val = pred.squeeze()
                            
                            if task_name in ['gate_a', 'gate_b', 'gate_c', 'gate_d', 'gate_e', 'ind_success', 'genotox']:
                                loss += F.binary_cross_entropy(pred_val, true)
                                
                                # Collect for accuracy
                                if task_name in val_preds:
                                    val_preds[task_name].extend(pred_val.cpu().numpy())
                                    val_trues[task_name].extend(true.cpu().numpy())
                            else:
                                loss += F.mse_loss(pred_val, true)
                    
                    val_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Calculate accuracy
            val_acc = {}
            for task in ['ind_success', 'gate_a', 'gate_c']:
                if val_preds[task]:
                    preds_binary = (np.array(val_preds[task]) > 0.5).astype(int)
                    trues_binary = (np.array(val_trues[task]) > 0.5).astype(int)
                    val_acc[task] = accuracy_score(trues_binary, preds_binary)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc.get('ind_success', 0))
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('models/nova_ind_best_model.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                print(f"  IND Success Acc: {val_acc.get('ind_success', 0):.3f}")
                print(f"  Gate A Acc: {val_acc.get('gate_a', 0):.3f}")
                print(f"  Gate C Acc: {val_acc.get('gate_c', 0):.3f}")
        
        return history
    
    def predict(self, smiles_list: list) -> pd.DataFrame:
        """예측"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        self.model.eval()
        
        # Create dummy targets
        dummy_targets = {k: np.zeros(len(smiles_list)) for k in ['gate_a', 'egfr_selectivity', 'ind_success']}
        
        dataset = MoleculeDatasetIND(smiles_list, dummy_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = {task: [] for task in self.model.task_heads.keys()}
        
        with torch.no_grad():
            for batch in loader:
                fps = batch['fingerprint'].to(self.device)
                outputs = self.model(fps)
                
                for task_name, preds in outputs.items():
                    preds_np = preds.cpu().numpy().flatten()
                    
                    # Denormalize continuous tasks
                    if task_name in self.scalers:
                        preds_np = self.scalers[task_name].inverse_transform(preds_np.reshape(-1, 1)).flatten()
                    
                    all_predictions[task_name].extend(preds_np)
        
        # Results DataFrame
        results = pd.DataFrame({
            'smiles': smiles_list,
            
            # Gates (probability)
            'pred_Gate_A_prob': all_predictions['gate_a'],
            'pred_Gate_B_prob': all_predictions['gate_b'],
            'pred_Gate_C_prob': all_predictions['gate_c'],
            'pred_Gate_D_prob': all_predictions['gate_d'],
            'pred_Gate_E_prob': all_predictions['gate_e'],
            'pred_IND_Success_prob': all_predictions['ind_success'],
            
            # Gates (binary)
            'pred_Gate_A': [p > 0.5 for p in all_predictions['gate_a']],
            'pred_Gate_B': [p > 0.5 for p in all_predictions['gate_b']],
            'pred_Gate_C': [p > 0.5 for p in all_predictions['gate_c']],
            'pred_Gate_D': [p > 0.5 for p in all_predictions['gate_d']],
            'pred_Gate_E': [p > 0.5 for p in all_predictions['gate_e']],
            'pred_IND_Success': [p > 0.5 for p in all_predictions['ind_success']],
            
            # Continuous metrics
            'pred_EGFR_Selectivity': all_predictions['egfr_selectivity'],
            'pred_hERG_IC50_uM': all_predictions['herg_ic50'],
            'pred_Solubility_uM': all_predictions['solubility'],
            'pred_NOAEL_Rat_mg_kg': all_predictions['noael_rat'],
            'pred_Starting_Dose_mg': all_predictions['starting_dose'],
            'pred_IND_Score': all_predictions['ind_score'],
        })
        
        # Risk Level
        def get_risk_level(score):
            if score >= 80:
                return "LOW"
            elif score >= 60:
                return "MEDIUM"
            elif score >= 40:
                return "HIGH"
            else:
                return "VERY HIGH"
        
        results['pred_Risk_Level'] = results['pred_IND_Score'].apply(get_risk_level)
        
        return results
    
    def save_model(self, path):
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'scalers': self.scalers,
        }, path)
        print(f"✅ Model saved: {path}")
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = INDGatePredictor()
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)
        
        self.scalers = checkpoint['scalers']
        print(f"✅ Model loaded: {path}")


def main():
    """메인 실행"""
    # Load IND gate data
    data_path = Path("generated_molecules/ind_gate_data.csv")
    
    if not data_path.exists():
        print("❌ IND gate data not found. Run nova_ind_gate_generator.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} molecules with IND gate data")
    
    # Train model
    print("\n🚀 Training IND Gate Predictor...")
    predictor = NOVAINDPredictor()
    history = predictor.train(df, epochs=50, batch_size=16, lr=0.001)
    
    print("\n✅ Training complete!")
    
    # Test prediction on Top 10
    print("\n🔮 Testing predictions on top 10 molecules...")
    top10_smiles = df['smiles'].head(10).tolist()
    predictions = predictor.predict(top10_smiles)
    
    print(predictions[['smiles', 'pred_IND_Success_prob', 'pred_IND_Score', 'pred_Risk_Level']].head(10))
    
    print(f"\n📈 Predicted IND Success rate: {predictions['pred_IND_Success'].sum() / len(predictions) * 100:.1f}%")


if __name__ == "__main__":
    main()

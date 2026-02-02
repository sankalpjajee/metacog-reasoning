"""
Ablation Study Framework for ACC-Inspired Probe (V4)

Tests the contribution of different feature groups:
1. Compressed hidden states (256 dims)
2. Dynamic features (9 dims)
3. Early branching entropy (2 dims)

Also tests different model configurations:
- Single probe vs ensemble
- Different hidden dimensions
- Different loss weights
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ACCProbeAblation(nn.Module):
    """ACC probe with configurable input dimensions for ablation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 64)
        
        self.wrong_head = nn.Linear(64, 1)
        self.conflict_head = nn.Linear(64, 1)
        self.utility_head = nn.Linear(64, 1)
        
        for layer in [self.fc1, self.fc2, self.wrong_head, self.conflict_head, self.utility_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        wrong_logits = self.wrong_head(x).squeeze(-1)
        conflict_score = torch.sigmoid(self.conflict_head(x)).squeeze(-1)
        utility_score = torch.tanh(self.utility_head(x)).squeeze(-1)
        
        return wrong_logits, conflict_score, utility_score


class AblationDataset(Dataset):
    """Dataset for ablation studies with feature masking."""
    def __init__(
        self,
        tensor_data: Dict[str, torch.Tensor],
        feature_mask: torch.Tensor = None,
    ):
        self.features = tensor_data['features']
        self.wrong_labels = tensor_data['wrong_labels']
        self.conflict_labels = tensor_data['conflict_labels']
        self.utility_labels = tensor_data['utility_labels']
        
        # Apply feature mask if provided
        if feature_mask is not None:
            self.features = self.features[:, feature_mask]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'wrong_label': self.wrong_labels[idx],
            'conflict_label': self.conflict_labels[idx],
            'utility_label': self.utility_labels[idx],
        }


def train_and_evaluate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    device: str,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    lambda_wrong: float = 1.0,
    lambda_conflict: float = 0.3,
    lambda_utility: float = 1.5,
) -> Dict:
    """Train and evaluate a single probe configuration."""
    
    model = ACCProbeAblation(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    mse_loss = nn.MSELoss()
    
    best_val_utility_corr = -1.0
    best_metrics = {}
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            wrong_labels = batch['wrong_label'].to(device)
            conflict_labels = batch['conflict_label'].to(device)
            utility_labels = batch['utility_label'].to(device)
            
            optimizer.zero_grad()
            
            wrong_logits, conflict_pred, utility_pred = model(features)
            
            loss_wrong = focal_loss(wrong_logits, wrong_labels)
            loss_conflict = mse_loss(conflict_pred, conflict_labels)
            loss_utility = mse_loss(utility_pred, utility_labels)
            
            total_loss = (
                lambda_wrong * loss_wrong +
                lambda_conflict * loss_conflict +
                lambda_utility * loss_utility
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        all_wrong_preds = []
        all_wrong_labels = []
        all_utility_preds = []
        all_utility_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                wrong_labels = batch['wrong_label'].to(device)
                utility_labels = batch['utility_label'].to(device)
                
                wrong_logits, _, utility_pred = model(features)
                
                wrong_probs = torch.sigmoid(wrong_logits)
                all_wrong_preds.extend(wrong_probs.cpu().numpy())
                all_wrong_labels.extend(wrong_labels.cpu().numpy())
                all_utility_preds.extend(utility_pred.cpu().numpy())
                all_utility_labels.extend(utility_labels.cpu().numpy())
        
        # Compute metrics
        wrong_preds_binary = (np.array(all_wrong_preds) > 0.5).astype(int)
        wrong_labels_binary = np.array(all_wrong_labels).astype(int)
        
        val_wrong_f1 = f1_score(wrong_labels_binary, wrong_preds_binary, zero_division=0)
        val_precision = precision_score(wrong_labels_binary, wrong_preds_binary, zero_division=0)
        val_recall = recall_score(wrong_labels_binary, wrong_preds_binary, zero_division=0)
        
        utility_corr, _ = pearsonr(all_utility_preds, all_utility_labels)
        if np.isnan(utility_corr):
            utility_corr = 0.0
        
        if utility_corr > best_val_utility_corr:
            best_val_utility_corr = utility_corr
            best_metrics = {
                'wrong_f1': val_wrong_f1,
                'wrong_precision': val_precision,
                'wrong_recall': val_recall,
                'utility_corr': utility_corr,
            }
    
    return best_metrics


def run_feature_ablation(
    data: Dict[str, torch.Tensor],
    device: str,
    compressed_dim: int = 256,
    dynamic_dim: int = 9,
    entropy_dim: int = 2,
    epochs: int = 20,
    batch_size: int = 64,
    val_split: float = 0.2,
) -> Dict:
    """Run feature ablation study."""
    
    total_dim = data['features'].shape[1]
    
    # Define feature groups
    feature_groups = {
        'all': list(range(total_dim)),
        'compressed_only': list(range(compressed_dim)),
        'dynamic_only': list(range(compressed_dim, compressed_dim + dynamic_dim)),
        'entropy_only': list(range(compressed_dim + dynamic_dim, total_dim)),
        'no_compressed': list(range(compressed_dim, total_dim)),
        'no_dynamic': list(range(compressed_dim)) + list(range(compressed_dim + dynamic_dim, total_dim)),
        'no_entropy': list(range(compressed_dim + dynamic_dim)),
    }
    
    results = {}
    
    for group_name, feature_indices in feature_groups.items():
        print(f"\n{'='*50}")
        print(f"Testing: {group_name} ({len(feature_indices)} features)")
        print(f"{'='*50}")
        
        # Create feature mask
        feature_mask = torch.tensor(feature_indices, dtype=torch.long)
        
        # Create dataset with mask
        dataset = AblationDataset(data, feature_mask)
        
        # Split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train and evaluate
        metrics = train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=len(feature_indices),
            hidden_dim=128,
            device=device,
            epochs=epochs,
        )
        
        results[group_name] = {
            'num_features': len(feature_indices),
            **metrics,
        }
        
        print(f"  Wrong F1: {metrics['wrong_f1']:.4f}")
        print(f"  Utility Corr: {metrics['utility_corr']:.4f}")
    
    return results


def run_architecture_ablation(
    data: Dict[str, torch.Tensor],
    device: str,
    epochs: int = 20,
    batch_size: int = 64,
    val_split: float = 0.2,
) -> Dict:
    """Run architecture ablation study."""
    
    input_dim = data['features'].shape[1]
    
    # Define configurations
    configs = [
        {'hidden_dim': 64, 'name': 'hidden_64'},
        {'hidden_dim': 128, 'name': 'hidden_128'},
        {'hidden_dim': 256, 'name': 'hidden_256'},
    ]
    
    results = {}
    
    # Create dataset
    dataset = AblationDataset(data)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"{'='*50}")
        
        metrics = train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            device=device,
            epochs=epochs,
        )
        
        results[config['name']] = {
            'hidden_dim': config['hidden_dim'],
            **metrics,
        }
        
        print(f"  Wrong F1: {metrics['wrong_f1']:.4f}")
        print(f"  Utility Corr: {metrics['utility_corr']:.4f}")
    
    return results


def run_loss_weight_ablation(
    data: Dict[str, torch.Tensor],
    device: str,
    epochs: int = 20,
    batch_size: int = 64,
    val_split: float = 0.2,
) -> Dict:
    """Run loss weight ablation study."""
    
    input_dim = data['features'].shape[1]
    
    # Define weight configurations
    configs = [
        {'lambda_wrong': 1.0, 'lambda_conflict': 0.0, 'lambda_utility': 0.0, 'name': 'wrong_only'},
        {'lambda_wrong': 0.0, 'lambda_conflict': 0.0, 'lambda_utility': 1.0, 'name': 'utility_only'},
        {'lambda_wrong': 1.0, 'lambda_conflict': 0.3, 'lambda_utility': 1.5, 'name': 'default'},
        {'lambda_wrong': 1.0, 'lambda_conflict': 0.5, 'lambda_utility': 2.0, 'name': 'high_utility'},
        {'lambda_wrong': 2.0, 'lambda_conflict': 0.3, 'lambda_utility': 1.0, 'name': 'high_wrong'},
    ]
    
    results = {}
    
    # Create dataset
    dataset = AblationDataset(data)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"  λ_wrong={config['lambda_wrong']}, λ_conflict={config['lambda_conflict']}, λ_utility={config['lambda_utility']}")
        print(f"{'='*50}")
        
        metrics = train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            hidden_dim=128,
            device=device,
            epochs=epochs,
            lambda_wrong=config['lambda_wrong'],
            lambda_conflict=config['lambda_conflict'],
            lambda_utility=config['lambda_utility'],
        )
        
        results[config['name']] = {
            'lambda_wrong': config['lambda_wrong'],
            'lambda_conflict': config['lambda_conflict'],
            'lambda_utility': config['lambda_utility'],
            **metrics,
        }
        
        print(f"  Wrong F1: {metrics['wrong_f1']:.4f}")
        print(f"  Utility Corr: {metrics['utility_corr']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation Study for ACC Probe (V4)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--compressed_dim", type=int, default=256)
    parser.add_argument("--dynamic_dim", type=int, default=9)
    parser.add_argument("--entropy_dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation_type", type=str, choices=['feature', 'architecture', 'loss', 'all'], default='all')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading training data...")
    all_features = []
    all_wrong_labels = []
    all_conflict_labels = []
    all_utility_labels = []
    
    for benchmark in args.benchmarks:
        tensor_path = os.path.join(args.data_dir, f"{benchmark}_tensors.pt")
        if not os.path.exists(tensor_path):
            print(f"Warning: {tensor_path} not found, skipping {benchmark}")
            continue
        
        print(f"Loading {benchmark}...")
        data = torch.load(tensor_path)
        all_features.append(data['features'])
        all_wrong_labels.append(data['wrong_labels'])
        all_conflict_labels.append(data['conflict_labels'])
        all_utility_labels.append(data['utility_labels'])
    
    combined_data = {
        'features': torch.cat(all_features, dim=0),
        'wrong_labels': torch.cat(all_wrong_labels, dim=0),
        'conflict_labels': torch.cat(all_conflict_labels, dim=0),
        'utility_labels': torch.cat(all_utility_labels, dim=0),
    }
    
    print(f"\nTotal samples: {len(combined_data['features'])}")
    print(f"Feature dimensions: {combined_data['features'].shape[1]}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    # Feature ablation
    if args.ablation_type in ['feature', 'all']:
        print("\n" + "="*60)
        print("FEATURE ABLATION STUDY")
        print("="*60)
        
        feature_results = run_feature_ablation(
            data=combined_data,
            device=device,
            compressed_dim=args.compressed_dim,
            dynamic_dim=args.dynamic_dim,
            entropy_dim=args.entropy_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
        )
        all_results['feature_ablation'] = feature_results
    
    # Architecture ablation
    if args.ablation_type in ['architecture', 'all']:
        print("\n" + "="*60)
        print("ARCHITECTURE ABLATION STUDY")
        print("="*60)
        
        arch_results = run_architecture_ablation(
            data=combined_data,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
        )
        all_results['architecture_ablation'] = arch_results
    
    # Loss weight ablation
    if args.ablation_type in ['loss', 'all']:
        print("\n" + "="*60)
        print("LOSS WEIGHT ABLATION STUDY")
        print("="*60)
        
        loss_results = run_loss_weight_ablation(
            data=combined_data,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
        )
        all_results['loss_weight_ablation'] = loss_results
    
    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    for study_name, study_results in all_results.items():
        print(f"\n{study_name.upper()}:")
        print("-" * 40)
        
        for config_name, metrics in study_results.items():
            print(f"  {config_name}:")
            print(f"    Wrong F1: {metrics['wrong_f1']:.4f}")
            print(f"    Utility Corr: {metrics['utility_corr']:.4f}")


if __name__ == "__main__":
    main()

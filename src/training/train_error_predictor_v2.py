"""
Enhanced Error Predictor Training (V2)

Trains error prediction probe with:
- 2-layer MLP architecture
- Ensemble of multiple probes
- Weighted focal loss
- Multi-feature input (hidden states + confidence + agreement)

Target accuracy: 80%+
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


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


class MLPProbe(nn.Module):
    """2-layer MLP probe for error prediction."""
    def __init__(
        self,
        hidden_dim: int = 16384,  # 4 layers * 4096 dims
        confidence_dim: int = 4,   # 4 confidence features
        agreement_dim: int = 1,    # 1 agreement rate
        intermediate_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        input_dim = hidden_dim + confidence_dim + agreement_dim
        
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, hidden_states, confidence_features, agreement_rates):
        # Concatenate all features
        x = torch.cat([hidden_states, confidence_features, agreement_rates.unsqueeze(-1)], dim=-1)
        
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits.squeeze(-1)


class EnsembleProbe(nn.Module):
    """Ensemble of multiple MLP probes."""
    def __init__(self, num_probes: int = 3, **probe_kwargs):
        super().__init__()
        self.probes = nn.ModuleList([MLPProbe(**probe_kwargs) for _ in range(num_probes)])
    
    def forward(self, hidden_states, confidence_features, agreement_rates):
        # Get predictions from all probes
        predictions = []
        for probe in self.probes:
            pred = probe(hidden_states, confidence_features, agreement_rates)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        return ensemble_pred


class ErrorPredictionDataset(Dataset):
    """Dataset for error prediction training."""
    def __init__(self, tensor_data: Dict[str, torch.Tensor]):
        self.hidden_states = tensor_data['hidden_states']
        self.labels = tensor_data['labels']
        self.confidence_features = tensor_data['confidence_features']
        self.agreement_rates = tensor_data['agreement_rates']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx],
            'confidence_features': self.confidence_features[idx],
            'agreement_rates': self.agreement_rates[idx],
        }


def train_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    use_focal_loss: bool = True,
) -> Dict:
    """Train the error prediction probe."""
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        # Compute class weights
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['labels'].tolist())
        num_correct = sum(1 for l in all_labels if l == 0)
        num_wrong = len(all_labels) - num_correct
        pos_weight = torch.tensor([num_correct / num_wrong]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_f1 = 0.0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            hidden_states = batch['hidden_states'].to(device)
            labels = batch['labels'].to(device)
            confidence_features = batch['confidence_features'].to(device)
            agreement_rates = batch['agreement_rates'].to(device)
            
            # Forward pass
            logits = model(hidden_states, confidence_features, agreement_rates)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                hidden_states = batch['hidden_states'].to(device)
                labels = batch['labels'].to(device)
                confidence_features = batch['confidence_features'].to(device)
                agreement_rates = batch['agreement_rates'].to(device)
                
                # Forward pass
                logits = model(hidden_states, confidence_features, agreement_rates)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Compute metrics
        accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        
        # Compute F1 score
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        history['val_accuracy'].append(accuracy)
        history['val_f1'].append(f1)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.4f}")
        print(f"  Val Precision: {precision:.4f}")
        print(f"  Val Recall: {recall:.4f}")
        print(f"  Val F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best F1: {f1:.4f}")
        
        # Step scheduler
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train enhanced error prediction probe")
    parser.add_argument("--data_dir", type=str, default="data/training/error_prediction_v2")
    parser.add_argument("--output_dir", type=str, default="models/error_predictor_v2")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble of probes")
    parser.add_argument("--num_probes", type=int, default=3, help="Number of probes in ensemble")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data from all benchmarks
    print("Loading training data...")
    all_tensor_data = {
        'hidden_states': [],
        'labels': [],
        'confidence_features': [],
        'agreement_rates': [],
    }
    
    for benchmark in args.benchmarks:
        tensor_path = os.path.join(args.data_dir, f"{benchmark}_tensors.pt")
        if not os.path.exists(tensor_path):
            print(f"Warning: {tensor_path} not found, skipping {benchmark}")
            continue
        
        tensor_data = torch.load(tensor_path)
        print(f"Loaded {benchmark}: {len(tensor_data['labels'])} samples")
        
        for key in all_tensor_data.keys():
            all_tensor_data[key].append(tensor_data[key])
    
    # Concatenate all data
    combined_data = {
        key: torch.cat(tensors, dim=0) for key, tensors in all_tensor_data.items()
    }
    
    print(f"\nTotal samples: {len(combined_data['labels'])}")
    num_correct = (combined_data['labels'] == 0).sum().item()
    num_wrong = (combined_data['labels'] == 1).sum().item()
    print(f"  Correct: {num_correct} ({num_correct/len(combined_data['labels'])*100:.1f}%)")
    print(f"  Wrong: {num_wrong} ({num_wrong/len(combined_data['labels'])*100:.1f}%)")
    
    # Create dataset
    dataset = ErrorPredictionDataset(combined_data)
    
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    if args.use_ensemble:
        print(f"Using ensemble of {args.num_probes} probes")
        model = EnsembleProbe(num_probes=args.num_probes)
    else:
        print("Using single MLP probe")
        model = MLPProbe()
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Train
    print("\nTraining...")
    history = train_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        use_focal_loss=True,
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "best_probe.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save config
    config = {
        'use_ensemble': args.use_ensemble,
        'num_probes': args.num_probes if args.use_ensemble else 1,
        'hidden_dim': 16384,
        'confidence_dim': 4,
        'agreement_dim': 1,
        'intermediate_dim': 256,
        'dropout': 0.1,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Save history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")
    
    # Print final results
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation F1: {max(history['val_f1']):.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

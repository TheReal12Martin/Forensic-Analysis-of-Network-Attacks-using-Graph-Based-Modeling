import warnings
from matplotlib import pyplot as plt
from numpy import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.graph_construction import build_graph_from_partition
from models.GAT import GAT
from config import Config
import gc
import os
from utils.monitoring import print_system_stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def _analyze_class_distribution(data, partition_idx):
    """Analyze and visualize class distribution"""
    # Move data to CPU for numpy operations
    train_counts = torch.bincount(data.y[data.train_mask].cpu())
    val_counts = torch.bincount(data.y[data.val_mask].cpu())
    
    imbalance_ratio = max(train_counts[0]/train_counts[1], 
                       train_counts[1]/train_counts[0])
    
    print(f"\n📊 Partition {partition_idx} Class Distribution:")
    print(f"- Train: Benign={train_counts[0].item()}, Malicious={train_counts[1].item()}")
    print(f"- Val:   Benign={val_counts[0].item()}, Malicious={val_counts[1].item()}")
    print(f"- Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > Config.IMBALANCE_THRESHOLD:
        print("⚠️ Warning: Severe class imbalance detected!")
    
    # Visualization (ensure matplotlib operations happen on CPU)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.bar(['Benign', 'Malicious'], train_counts.numpy())
    plt.title('Train Set Distribution')
    
    plt.subplot(122)
    plt.bar(['Benign', 'Malicious'], val_counts.numpy())
    plt.title('Validation Set Distribution')
    
    plt.savefig(f'class_dist_partition_{partition_idx}.png')
    plt.close()
    
    return imbalance_ratio

def _calculate_metrics(model, data, mask, device, prefix=""):
    """Calculate metrics for a specific mask"""
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        
        # Move predictions to CPU for sklearn metrics
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = preds[mask]
        y_probs = probs[mask, 1]

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_probs),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=['Benign', 'Malicious'],
                output_dict=True
            )
        }

import copy
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

def train_model(data, device, partition_idx=None):
    """Enhanced training function with detailed metrics logging"""
    # ======================
    # 1. Initial Setup
    # ======================
    data = data.to(device)
    print(f"\n=== Training Partition {partition_idx} on {device} ===")
    
    # Initialize global best tracking if not exists
    if not hasattr(Config, 'GLOBAL_BEST_ACC'):
        Config.GLOBAL_BEST_ACC = 0
        Config.GLOBAL_BEST_STATE = None
        Config.GLOBAL_BEST_METRICS = None
        Config.GLOBAL_BEST_PARTITION = None

    # ======================
    # 2. Model Configuration
    # ======================
    model = GAT(num_features=data.num_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=Config.SCHEDULER_PATIENCE,
        threshold=Config.SCHEDULER_THRESHOLD
    )

    class_weights = torch.tensor(Config.CLASS_WEIGHTS, device=device)

    # ======================
    # 3. Training Loop
    # ======================
    best_val_acc = 0
    best_model_state = None
    early_stop_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Loss calculation
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        if epoch % 2 == 0 or epoch == Config.EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                y_true = data.y[data.val_mask].cpu().numpy()
                y_pred = logits.argmax(dim=1)[data.val_mask].cpu().numpy()
                y_probs = torch.softmax(logits, dim=1)[data.val_mask, 1].cpu().numpy()

                val_acc = accuracy_score(y_true, y_pred)
                val_f1 = f1_score(y_true, y_pred)
                
                # Update best model for this partition
                if val_acc > best_val_acc + Config.MIN_DELTA:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}  # Proper handling of tensors
                    torch.save(best_model_state, f'best_model_part_{partition_idx}.pt')
                    
                    # Update global best if needed
                    if val_acc > Config.GLOBAL_BEST_ACC + Config.MIN_DELTA:
                        Config.GLOBAL_BEST_ACC = val_acc
                        Config.GLOBAL_BEST_STATE = best_model_state
                        Config.GLOBAL_BEST_PARTITION = partition_idx
                        torch.save({
                            'state_dict': best_model_state,
                            'val_accuracy': val_acc,
                            'partition': partition_idx,
                            'config': {
                                'num_features': data.num_features,
                                'hidden_channels': Config.HIDDEN_CHANNELS,
                                'heads': Config.HEADS,
                                'num_layers': Config.GAT_LAYERS
                            }
                        }, 'best_model.pt')
                else:
                    early_stop_counter += 1

                # Early stopping
                if early_stop_counter >= Config.PATIENCE:
                    break

    # ======================
    # 4. Final Evaluation
    # ======================
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Calculate final metrics
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        y_true = data.y.cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_probs = torch.softmax(logits, dim=1).cpu().numpy()

    metrics = {
        'val': {
            'accuracy': accuracy_score(y_true[data.val_mask.cpu().numpy()], y_pred[data.val_mask.cpu().numpy()]),
            'f1': f1_score(y_true[data.val_mask.cpu().numpy()], y_pred[data.val_mask.cpu().numpy()]),
            'recall': recall_score(y_true[data.val_mask.cpu().numpy()], y_pred[data.val_mask.cpu().numpy()]),
            'precision': precision_score(y_true[data.val_mask.cpu().numpy()], y_pred[data.val_mask.cpu().numpy()]),
            'roc_auc': roc_auc_score(y_true[data.val_mask.cpu().numpy()], y_probs[data.val_mask.cpu().numpy(), 1])
        }
    }

    if hasattr(data, 'test_mask'):
        metrics['test'] = {
            'accuracy': accuracy_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()]),
            'f1': f1_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()]),
            'recall': recall_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()]),
            'precision': precision_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()]),
            'roc_auc': roc_auc_score(y_true[data.test_mask.cpu().numpy()], y_probs[data.test_mask.cpu().numpy(), 1])
        }

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return model, best_val_acc, metrics

def smooth_loss(pred, target, weight=None, smoothing=0.1):
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        if weight is not None:
            loss = loss * weight[target]
        return loss.mean()

import warnings
from matplotlib import pyplot as plt
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
    train_counts = torch.bincount(data.y[data.train_mask])
    val_counts = torch.bincount(data.y[data.val_mask])
    
    imbalance_ratio = max(train_counts[0]/train_counts[1], 
                         train_counts[1]/train_counts[0])
    
    print(f"\n📊 Partition {partition_idx} Class Distribution:")
    print(f"- Train: Benign={train_counts[0]}, Malicious={train_counts[1]}")
    print(f"- Val:   Benign={val_counts[0]}, Malicious={val_counts[1]}")
    print(f"- Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > Config.IMBALANCE_THRESHOLD:
        print("⚠️ Warning: Severe class imbalance detected!")
    
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.bar(['Benign', 'Malicious'], train_counts.cpu().numpy())
    plt.title('Train Set Distribution')
    
    plt.subplot(122)
    plt.bar(['Benign', 'Malicious'], val_counts.cpu().numpy())
    plt.title('Validation Set Distribution')
    
    plt.savefig(f'class_dist_partition_{partition_idx}.png')
    plt.close()
    
    return imbalance_ratio

def train_model(data, device, partition_idx=None):
    """Enhanced training with separate validation/test metrics"""
    # Dynamic class weighting

    imbalance_ratio = _analyze_class_distribution(data, partition_idx)
    
    # Dynamic weight adjustment based on imbalance
    if imbalance_ratio > 5.0:
        Config.CLASS_WEIGHTS[1] = min(20.0, Config.CLASS_WEIGHTS[1] * 1.5)
        print(f"⚠️ Adjusting class weights to: {Config.CLASS_WEIGHTS}")

    val_metrics_history = []
    class_counts = torch.bincount(data.y)
    weights = (torch.sqrt(1. / class_counts.float())).to(device)
    
    # Focal loss
    def focal_loss(inputs, targets, alpha=0.75, gamma=2.0):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (alpha * (1-pt)**gamma * BCE_loss).mean()
    
    # Model setup
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
        threshold=Config.SCHEDULER_THRESHOLD,
    )

    # Training loop
    best_val_acc = 0
    best_val_metrics = None
    early_stop_counter = 0
    
    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = focal_loss(
            out[data.train_mask], 
            data.y[data.train_mask],
            alpha=Config.FOCAL_ALPHA,
            gamma=Config.FOCAL_GAMMA
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation phase
        if epoch % 2 == 0:
            val_metrics = _calculate_metrics(
                model, data, data.val_mask, device, "val"
            )
            val_metrics_history.append(val_metrics)
            
            scheduler.step(val_metrics['accuracy'])
            
            if val_metrics['accuracy'] > best_val_acc + Config.MIN_DELTA:
                best_val_acc = val_metrics['accuracy']
                early_stop_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
                best_val_metrics = val_metrics
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Calculate final metrics
    metrics = {
        'val': best_val_metrics,
        'test': None
    }
    
    if sum(data.test_mask) >= Config.MIN_TEST_SAMPLES:
        metrics['test'] = _calculate_metrics(
            model, data, data.test_mask, device, "test"
        )
    
    # Print comprehensive report
    _print_metrics_report(metrics, partition_idx)
    
    return model, best_val_acc, metrics

def _calculate_metrics(model, data, mask, device, prefix=""):
    """Calculate metrics for a specific mask"""
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
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

def _print_metrics_report(metrics, partition_idx):
    """Print formatted metrics report"""
    if metrics['val']:
        print(f"\n⭐ Validation Metrics (Partition {partition_idx}):")
        print(f"- Accuracy:  {metrics['val']['accuracy']:.4f}")
        print(f"- F1:       {metrics['val']['f1']:.4f}")
        print(f"- Recall:   {metrics['val']['recall']:.4f}")
        print(f"- Precision:{metrics['val']['precision']:.4f}")
        print(f"- ROC AUC:  {metrics['val']['roc_auc']:.4f}")
    
    if metrics['test']:
        print(f"\n📊 Test Metrics (Partition {partition_idx}):")
        print(f"- Accuracy:  {metrics['test']['accuracy']:.4f}")
        print(f"- F1:       {metrics['test']['f1']:.4f}")
        print(f"- Recall:   {metrics['test']['recall']:.4f}")
        print(f"- Precision:{metrics['test']['precision']:.4f}")
        print(f"- ROC AUC:  {metrics['test']['roc_auc']:.4f}")
import warnings
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.graph_construction import build_graph_from_partition
from models.GAT import GAT
from config import Config
import gc
import os
from utils.monitoring import print_system_stats
from utils.pyg_conversion import convert_to_pyg_memory_safe
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_model(data, device):
    # Dynamic class weighting
    class_counts = torch.bincount(data.y)
    weights = (torch.sqrt(1. / class_counts.float())).to(device)  # Square root weighting
    
    # Focal loss
    def focal_loss(inputs, targets, alpha=0.75, gamma=2.0):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (alpha * (1-pt)**gamma * BCE_loss).mean()
        return focal_loss
    
    # Model and optimizer
    model = GAT(num_features=data.num_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=Config.LEARNING_RATE, 
                                weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5, factor=0.5)
    
    best_val_acc = 0
    early_stop_counter = 0
    
    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = focal_loss(out[data.train_mask], 
                         data.y[data.train_mask],
                         alpha=Config.FOCAL_ALPHA,
                         gamma=Config.FOCAL_GAMMA)
        
        # Gradient clipping and accumulation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                
                scheduler.step(val_acc)  # Update LR
                
                if val_acc > best_val_acc + Config.MIN_DELTA:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                    torch.save(model.state_dict(), 'best_model.pt')
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= Config.PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model, best_val_acc
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from models.GAT import GAT
from config import Config
from utils.monitoring import log_resource_usage
import gc

def train_model(data, device):
    """Train with cross-validation and class balancing"""
    try:
        # Calculate class weights
        class_counts = torch.bincount(data.y)
        weights = (1. / class_counts.float()).to(device)
        
        # K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(range(data.num_nodes), data.y)):
            log_resource_usage(f"Fold {fold+1} start")
            
            # Create masks
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = True
            data.test_mask[test_idx] = True
            
            # Model and optimizer
            model = GAT(num_features=data.num_features).to(device)
            optimizer = torch.optim.Adam(model.parameters(), 
                                      lr=Config.LEARNING_RATE,
                                      weight_decay=Config.WEIGHT_DECAY)
            
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(Config.EPOCHS):
                model.train()
                optimizer.zero_grad()
                
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], 
                                 data.y[data.train_mask],
                                 weight=weights)
                
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred = model(data.x, data.edge_index).argmax(dim=1)
                        val_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
                        
                        if val_acc > best_val_acc + Config.MIN_DELTA:
                            best_val_acc = val_acc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= Config.PATIENCE:
                            break
                
                gc.collect()
            
            results.append(best_val_acc)
            log_resource_usage(f"Fold {fold+1} end")
        
        return model, sum(results)/len(results)
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None, 0
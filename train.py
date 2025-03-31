import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models.GAT import GAT
from config import Config
import gc

from utils.graph_construction import build_graph_from_partition
from utils.pyg_conversion import convert_to_pyg_memory_safe

def train_partition(partition_file):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Graph Construction
        graph, ip_to_idx = build_graph_from_partition(partition_file)
        
        # 2. Convert to PyG
        data = convert_to_pyg_memory_safe(graph, device)
        del graph
        gc.collect()
        
        # 3. Train GAT
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            heads=Config.HEADS,
            num_layers=Config.GAT_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=Config.LEARNING_RATE,
                                    weight_decay=Config.WEIGHT_DECAY)
        
        # Class weights
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, device=device)
        
        best_val_acc = 0
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], 
                                 data.y[data.train_mask],
                                 weight=class_weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                model.eval()
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model.state_dict()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        return model, data
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
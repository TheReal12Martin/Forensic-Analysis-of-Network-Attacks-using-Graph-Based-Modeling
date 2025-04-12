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


def train_partition(partition_file):
    """Train on a single partition with strict resource control"""
    try:
        # Apply CPU/memory limits
        Config.apply_cpu_limits()
        print_system_stats("Training starting")
        
        device = torch.device('cpu')
        
        # Build graph
        graph, _ = build_graph_from_partition(partition_file)
        if graph is None:
            return None, None
            
        # Convert to PyG
        data = convert_to_pyg_memory_safe(graph, device)
        del graph
        gc.collect()
        
        if data is None:
            return None, None

        # Verify data types
        data.y = data.y.long()
        if data.y.dtype != torch.long:
            raise ValueError(f"Labels must be long dtype, got {data.y.dtype}")

        # Minimal model
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            heads=Config.HEADS,
            num_layers=Config.GAT_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=Config.LEARNING_RATE
        )
        
        best_weights = None
        best_val_acc = 0
        
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Memory cleanup - always exists
            del out
            gc.collect()
            
            # Validation (only creates pred when needed)
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(data.x, data.edge_index).argmax(dim=1)
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_weights = model.state_dict().copy()
                
                # Clean validation tensors
                del pred
                print_system_stats(f"Epoch {epoch}")
                gc.collect()

        # Load best weights
        model.load_state_dict(best_weights)
        return model, data

    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None, None
    finally:
        gc.collect()
        print_system_stats("Training completed")
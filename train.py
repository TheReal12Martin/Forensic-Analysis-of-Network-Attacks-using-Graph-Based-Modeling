import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from models.GAT import GAT
from utils.graph_construction import build_graph_from_partition
from utils.pyg_conversion import convert_to_pyg_memory_safe
from config import Config
import gc
import os
import psutil

def print_memory_usage():
    """Helper function to monitor memory usage"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # In MB
    print(f"Memory usage: {mem:.2f} MB")

def train_partition(partition_file):
    """Train GAT model on a single data partition"""
    try:
        print(f"\n=== Training on Partition: {os.path.basename(partition_file)} ===")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # ========== Graph Construction ==========
        print("\n[1/4] Building graph from partition...")
        print_memory_usage()
        graph, ip_mapping = build_graph_from_partition(partition_file)
        print_memory_usage()
        
        # ========== PyG Conversion ==========
        print("\n[2/4] Converting to PyTorch Geometric format...")
        data = convert_to_pyg_memory_safe(graph, device)
        
        del graph  # Free memory
        gc.collect()
        print_memory_usage()
        
        # ========== Train/Val/Test Split ==========
        print("\n[3/4] Creating dataset splits...")
        node_indices = np.arange(data.num_nodes)
        node_labels = data.y.cpu().numpy()
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            node_indices,
            test_size=Config.TEST_RATIO,
            stratify=node_labels,
            random_state=Config.RANDOM_STATE
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=Config.VAL_RATIO/(1-Config.TEST_RATIO),
            stratify=node_labels[train_val_idx],
            random_state=Config.RANDOM_STATE
        )
        
        # Create masks
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        # ========== Model Training ==========
        print("\n[4/4] Training GAT model...")
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            heads=Config.HEADS,
            dropout=Config.DROPOUT,
            num_layers=Config.GAT_LAYERS
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        no_improve = 0
        
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(
                out[data.train_mask],
                data.y[data.train_mask],
                weight=torch.tensor(Config.CLASS_WEIGHTS, device=device)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                
                # Early stopping check
                if val_loss < best_val_loss - Config.MIN_DELTA:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(model.state_dict(), 'best_model.pt')
                else:
                    no_improve += 1
            
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            if no_improve >= Config.PATIENCE:
                print("Early stopping triggered")
                break
            
            # Periodic memory cleanup
            if epoch % 10 == 0:
                gc.collect()
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pt'))
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
            print(f"\nTest Accuracy: {test_acc:.4f}")
        
        return model, data
        
    except Exception as e:
        print(f"\nTraining failed on partition {partition_file}: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'data' in locals():
            del data
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
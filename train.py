import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from models.GAT import GAT
from utils.graph_construction import build_graph_from_partition
from utils.pyg_conversion import convert_to_pyg_memory_safe
from utils.losses import FocalLoss, weighted_loss
from config import Config
import torch.nn as nn
import gc
import os
import psutil
import sys
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Memory usage: {mem:.2f} MB")

def visualize_subgraph(data, num_nodes=100):
    """Visualize a small subgraph for debugging"""
    try:
        print("\nVisualizing subgraph...")
        if num_nodes > data.num_nodes:
            num_nodes = data.num_nodes
        
        # Create subgraph
        subset = torch.arange(num_nodes, device=data.x.device)
        sub_data = data.subgraph(subset)
        
        # Convert to networkx
        g = to_networkx(sub_data, to_undirected=True)
        
        # Draw
        plt.figure(figsize=(12, 12))
        nx.draw(g, with_labels=True, node_size=200, font_size=8)
        plt.title(f"Subgraph of first {num_nodes} nodes")
        plt.savefig("subgraph_debug.png")
        plt.close()
        print("Saved subgraph visualization to subgraph_debug.png")
    except Exception as e:
        print(f"Failed to visualize subgraph: {str(e)}")

def check_device_consistency(data):
    """Verify all tensors are on the same device"""
    devices = [t.device for t in [
        data.x, 
        data.edge_index, 
        data.y,
        getattr(data, 'train_mask', torch.tensor(0)),
        getattr(data, 'val_mask', torch.tensor(0)),
        getattr(data, 'test_mask', torch.tensor(0))
    ]]
    
    unique_devices = set(devices)
    if len(unique_devices) > 1:
        raise ValueError(f"Mixed devices detected: {unique_devices}")
    return devices[0]

def debug_data_object(data):
    """Print comprehensive debug information about the PyG Data object"""
    print("\n=== DATA OBJECT DEBUG INFO ===")
    print(f"Data object keys: {data.keys}")
    print(f"Number of nodes: {data.num_nodes}")
    
    # Tensor shapes and devices
    print("\nTensor properties:")
    for key in ['x', 'edge_index', 'y', 'train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, key):
            tensor = getattr(data, key)
            print(f"{key}: shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}")
    
    # Edge index validation
    print("\nEdge index validation:")
    print(f"Min edge index: {data.edge_index.min().item()}")
    print(f"Max edge index: {data.edge_index.max().item()}")
    print(f"Num nodes: {data.num_nodes}")
    
    if data.edge_index.max() >= data.num_nodes:
        print("ERROR: Edge index contains invalid node references!")
    
    # Feature statistics
    print("\nFeature statistics:")
    print(f"NaN values in x: {torch.isnan(data.x).any().item()}")
    print(f"Inf values in x: {torch.isinf(data.x).any().item()}")
    print(f"Feature mean: {data.x.mean().item():.4f}")
    print(f"Feature std: {data.x.std().item():.4f}")
    
    # Label distribution
    if hasattr(data, 'y'):
        unique, counts = torch.unique(data.y, return_counts=True)
        print("\nLabel distribution:")
        for u, c in zip(unique, counts):
            print(f"Class {u.item()}: {c.item()} samples")

def train_partition(partition_file):
    try:
        print(f"\n=== Training on Partition: {os.path.basename(partition_file)} ===")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print_memory_usage()
        
        # ===== 1. Graph Construction =====
        print("\n[1/4] Building graph from partition...")
        graph, ip_mapping = build_graph_from_partition(partition_file)
        
        # Basic graph stats
        print("\nGraph statistics:")
        print(f"Number of nodes: {len(graph.nodes)}")
        print(f"Number of edges: {len(graph.edges)}")
        print(f"Sample node features: {graph.nodes[0]['features'].shape}")
        print_memory_usage()
        
        # ===== 2. PyG Conversion =====
        print("\n[2/4] Converting with strict validation...")
        data = convert_to_pyg_memory_safe(graph, device)
        del graph  # Free memory
        gc.collect()
        print_memory_usage()
        
        # Debug data object
        debug_data_object(data)
        
        # Visualize a small subgraph
        visualize_subgraph(data, num_nodes=100)
        
        # ===== 3. Dataset Splits =====
        print("\n[3/4] Creating dataset splits...")
        
        # Verify device consistency before splitting
        check_device_consistency(data)
        
        # Convert to numpy on CPU for sklearn
        node_indices = torch.arange(data.num_nodes, device='cpu')
        node_labels = data.y.cpu().numpy()
        
        # Create splits
        train_val_idx, test_idx = train_test_split(
            node_indices.numpy(),
            test_size=Config.TEST_RATIO,
            stratify=node_labels,
            random_state=Config.RANDOM_STATE
        )
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=Config.VAL_RATIO/(1-Config.TEST_RATIO),
            stratify=node_labels[train_val_idx],
            random_state=Config.RANDOM_STATE
        )
        
        # Create masks on the correct device
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        # Debug splits
        print("\nSplit sizes:")
        print(f"Training: {data.train_mask.sum().item()}")
        print(f"Validation: {data.val_mask.sum().item()}")
        print(f"Test: {data.test_mask.sum().item()}")
        print_memory_usage()
        
        # ===== 4. Model Training =====
        print("\n[4/4] Training GAT model...")
        
        # Verify feature dimensions
        print("\n=== Dimension Verification ===")
        print(f"Data contains {data.num_nodes} nodes with {data.num_features} features")
        print(f"First node features shape: {data.x[0].shape}")
        
        # Update config if needed
        Config.update_feature_dimensions(data.num_features)
        
        # Initialize model
        model = GAT(num_features=Config.INPUT_FEATURES).to(device)
        
        # Handle class imbalance
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Training loop
        best_val_f1 = 0
        no_improve = 0
        best_model_state = None
        
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                
                pred = out.argmax(dim=1)
                report = classification_report(
                    data.y[data.val_mask].cpu().numpy(),
                    pred[data.val_mask].cpu().numpy(),
                    target_names=Config.CLASS_NAMES,
                    output_dict=True,
                    zero_division=0
                )
                val_f1 = report['weighted avg']['f1-score']  # Changed to weighted avg
                
                # Track best model
                if val_f1 > best_val_f1 + Config.MIN_DELTA:
                    best_val_f1 = val_f1
                    no_improve = 0
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
            
            # Print metrics for both classes
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | '
                  f'Benign F1: {report["Benign"]["f1-score"]:.4f} | '
                  f'Malicious F1: {report["Malicious"]["f1-score"]:.4f}')
            
            if no_improve >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            print("\n=== Final Test Results ===")
            print(classification_report(
                data.y[data.test_mask].cpu().numpy(),
                pred[data.test_mask].cpu().numpy(),
                target_names=Config.CLASS_NAMES,
                digits=4
            ))
            
            # Confusion matrix with percentages
            cm = confusion_matrix(
                data.y[data.test_mask].cpu().numpy(),
                pred[data.test_mask].cpu().numpy()
            )
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("\nConfusion Matrix (Counts):")
            print(cm)
            print("\nConfusion Matrix (Percentages):")
            print(np.round(cm_percent, 3))
        
        return model, data
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}", file=sys.stderr)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
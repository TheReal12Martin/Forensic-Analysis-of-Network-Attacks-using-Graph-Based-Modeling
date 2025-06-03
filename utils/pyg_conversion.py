import torch
from torch_geometric.data import Data
import numpy as np
from config import Config
from sklearn.model_selection import train_test_split
import gc

def convert_to_pyg_memory_safe(graph, device):
    """Convert networkx graph to PyG Data with strict validation"""
    if graph is None:
        print("No graph provided")
        return None
        
    num_nodes = graph.number_of_nodes()
    if num_nodes < Config.MIN_GRAPH_NODES:
        print(f"Graph too small ({num_nodes} < {Config.MIN_GRAPH_NODES} nodes)")
        return None

    try:
        nodes = list(graph.nodes())
        
        # Feature matrix with duplicate checking
        unique_features = []
        unique_indices = []
        feature_set = set()
        
        for n in nodes:
            feat = tuple(graph.nodes[n]['features'].astype(np.float32).round(4))
            if feat not in feature_set:
                feature_set.add(feat)
                unique_features.append(graph.nodes[n]['features'])
                unique_indices.append(n)
        
        if len(unique_features) < num_nodes:
            print(f"Removed {num_nodes - len(unique_features)} duplicates during conversion")
            num_nodes = len(unique_features)
        
        x = torch.stack([torch.from_numpy(feat) for feat in unique_features])
        
        # Labels
        y = torch.tensor(
            [graph.nodes[n]['label'] for n in unique_indices],
            dtype=torch.long
        )
        
        # Edge processing with relabeling
        edge_index = []
        edge_attr = []
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        for u, v, data in graph.edges(data=True):
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                edge_attr.append(data.get('weight', 1.0))
        
        if not edge_index:
            print("No valid edges in graph")
            return None
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Stratified splitting with strict no-overlap
        indices = torch.arange(num_nodes)
        labels = y.cpu().numpy()
        
        # First split: train vs temp (val+test)
        idx_train, idx_temp = train_test_split(
            indices,
            train_size=0.6,
            stratify=labels,
            random_state=Config.RANDOM_STATE
        )
        
        # Second split: val vs test
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=0.5,
            stratify=labels[idx_temp],
            random_state=Config.RANDOM_STATE
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True
        
        # Verify minimum sizes
        if (sum(train_mask) < Config.MIN_TRAIN_SAMPLES or 
            sum(val_mask) < Config.MIN_VAL_SAMPLES or
            sum(test_mask) < Config.MIN_TEST_SAMPLES):
            print("Insufficient samples after splitting")
            return None
            
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        return data.to(device)
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return None
    finally:
        gc.collect()
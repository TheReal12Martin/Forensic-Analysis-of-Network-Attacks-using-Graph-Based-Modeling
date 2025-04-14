import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from config import Config
from sklearn.model_selection import train_test_split
import gc

def convert_to_pyg_memory_safe(graph, device):
    if graph is None:
        print("No graph provided")
        return None
        
    num_nodes = graph.number_of_nodes()
    if num_nodes < Config.MIN_GRAPH_NODES:
        print(f"Graph too small ({num_nodes} < {Config.MIN_GRAPH_NODES} nodes)")
        return None

    try:
        nodes = list(graph.nodes())
        
        # Feature matrix
        x = torch.stack([
            torch.from_numpy(graph.nodes[n]['features'].astype(np.float32)) 
            for n in nodes
        ])
        
        # Labels
        y = torch.tensor(
            [graph.nodes[n]['label'] for n in nodes],
            dtype=torch.long
        )
        
        # Edge processing
        edge_index, edge_attr = [], []
        for u, v, data in graph.edges(data=True):
            edge_index.append([u, v])
            edge_attr.append(data.get('weight', 1.0))
        
        if not edge_index:  # Handle empty graphs
            print("No edges in graph")
            return None
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Train/val/test split with minimum sizes
        num_train = max(int(0.6 * num_nodes), Config.MIN_TRAIN_SAMPLES)
        num_val = max(int(0.2 * num_nodes), Config.MIN_VAL_SAMPLES)
        
        if num_nodes < (num_train + num_val + Config.MIN_TEST_SAMPLES):
            print(f"Insufficient nodes ({num_nodes}) for proper splitting")
            return None
            
        indices = torch.randperm(num_nodes)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            train_mask=indices[:num_train],
            val_mask=indices[num_train:num_train+num_val],
            test_mask=indices[num_train+num_val:]
        )
        
        return data.to(device)
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return None
    finally:
        gc.collect()
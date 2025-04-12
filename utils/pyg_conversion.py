import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from config import Config
from sklearn.model_selection import train_test_split
import gc

def convert_to_pyg_memory_safe(graph, device):
    try:
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        # Process features
        x = torch.stack([
            torch.from_numpy(graph.nodes[n]['features'].astype(np.float32)) 
            for n in nodes
        ])
        
        # Convert labels to int64 (Long) explicitly
        y = torch.tensor(
            [graph.nodes[n]['label'] for n in nodes],
            dtype=torch.long  # Explicitly set to long
        )
        
        # Process edges
        edge_index = []
        edge_attr = []
        for u, v, data in graph.edges(data=True):
            edge_index.append([u, v])
            edge_attr.append(data.get('weight', 1.0))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Create masks
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,  # Now properly typed as long
            train_mask=indices[:train_size],
            val_mask=indices[train_size:train_size+val_size],
            test_mask=indices[train_size+val_size:]
        )
        
        return data.to(device)
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return None
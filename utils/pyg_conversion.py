import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from config import Config
import gc

def convert_to_pyg_memory_safe(graph, device='cpu'):
    try:
        nodes = sorted(graph.nodes())
        num_nodes = len(nodes)
        
        # Initialize tensors
        x = torch.zeros((num_nodes, len(graph.nodes[0]['features'])), dtype=torch.float32)
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        # Process nodes in chunks
        chunk_size = 50000
        for i in range(0, num_nodes, chunk_size):
            chunk_nodes = nodes[i:i+chunk_size]
            for n in chunk_nodes:
                x[n] = torch.from_numpy(graph.nodes[n]['features'])
                y[n] = graph.nodes[n].get('label', 0)
        
        # Process edges
        edges = list(graph.edges(data='weight'))
        edge_index = torch.zeros((2, len(edges)*2), dtype=torch.long)
        edge_attr = torch.zeros(len(edges)*2, dtype=torch.float32)
        
        for i, (u, v, w) in enumerate(tqdm(edges, desc="Processing edges")):
            edge_index[:, i*2] = torch.tensor([u, v])
            edge_index[:, i*2+1] = torch.tensor([v, u])
            edge_attr[i*2] = edge_attr[i*2+1] = w if w is not None else 1.0
        
        # Create train/val/test masks
        num_nodes = len(y)
        indices = torch.randperm(num_nodes)
        
        train_size = int(num_nodes * (1 - Config.TEST_RATIO - Config.VAL_RATIO))
        val_size = int(num_nodes * Config.VAL_RATIO)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        # Create and validate Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes
        )
        
        # DEBUG: Verify class balance (now correctly placed after data creation)
        print(f"🔍 Full dataset class balance: {torch.unique(data.y, return_counts=True)}")
        print(f"🔍 Test set samples: {test_mask.sum().item()}")
        
        # Validate before moving to device
        if data.edge_index.max() >= data.num_nodes:
            raise ValueError("Invalid edge indices detected")
        
        # Move to device
        data = data.to(device)
        print(f"Successfully moved data to {device}")
        
        return data
        
    except Exception as e:
        print(f"PyG conversion failed: {str(e)}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
import torch
from torch_geometric.data import Data
import numpy as np
from config import Config
import gc

def convert_to_pyg_memory_safe(graph, device='cpu'):
    try:
        print("\nStarting PyG conversion with enhanced validation...")
        nodes = sorted(graph.nodes())
        num_nodes = len(nodes)
        
        # Critical validation - must have contiguous 0-based indices
        if num_nodes > 0 and nodes != list(range(num_nodes)):
            raise ValueError(
                f"Node indices must be exactly 0 to {num_nodes-1}. "
                f"Found indices: min={min(nodes)}, max={max(nodes)}"
            )

        # Initialize tensors
        feature_size = len(graph.nodes[0]['features'])
        x = torch.zeros((num_nodes, feature_size), dtype=torch.float32)
        y = torch.zeros(num_nodes, dtype=torch.long)

        # Load node data
        for idx in nodes:
            x[idx] = torch.from_numpy(graph.nodes[idx]['features'])
            y[idx] = graph.nodes[idx]['label']

        # Process edges
        edges = list(graph.edges(data='weight'))
        edge_index = torch.zeros((2, len(edges)*2), dtype=torch.long)
        edge_attr = torch.zeros(len(edges)*2, dtype=torch.float32)

        for i, (u, v, w) in enumerate(edges):
            edge_index[:, i*2] = torch.tensor([u, v])
            edge_index[:, i*2+1] = torch.tensor([v, u])
            edge_attr[i*2] = edge_attr[i*2+1] = w if w is not None else 1.0

        print("\nConversion validation:")
        print(f"Node feature matrix shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Max edge index: {edge_index.max().item()}")
        print(f"Min edge index: {edge_index.min().item()}")
        print(f"Actual num nodes in graph: {num_nodes}")

        # Create Data object on CPU first
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )

        # Verify data integrity before moving to device
        if data.edge_index.max() >= data.num_nodes:
            raise ValueError(
                f"Edge index contains invalid references! "
                f"Max index: {data.edge_index.max().item()}, "
                f"Num nodes: {data.num_nodes}"
            )

        # Move to device only after validation
        data = data.to(device)
        print(f"Successfully moved data to {device}")
        
        return data
        
    except Exception as e:
        print(f"PyG conversion failed: {str(e)}")
        raise
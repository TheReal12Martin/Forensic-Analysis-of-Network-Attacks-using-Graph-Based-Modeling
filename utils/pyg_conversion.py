import torch
from torch_geometric.data import Data
import numpy as np
import gc

def convert_to_pyg_memory_safe(graph, device='cpu'):
    try:
        print("Converting with memory safety...")
        
        # Validate graph structure
        if len(graph.nodes()) == 0:
            raise ValueError("Graph contains no nodes")
        
        # Process nodes with validation
        nodes = list(graph.nodes())
        num_features = len(graph.nodes[nodes[0]]['features']) if nodes else 0
        
        # Initialize arrays with default values
        x = torch.zeros((len(nodes), num_features), dtype=torch.float32)
        y = torch.zeros(len(nodes), dtype=torch.long)
        
        for i, n in enumerate(nodes):
            # Handle missing features
            node_features = graph.nodes[n].get('features')
            if node_features is None:
                print(f"Warning: Node {n} has no features, using zeros")
                node_features = np.zeros(num_features, dtype=np.float32)
            
            # Handle missing labels
            node_label = graph.nodes[n].get('label', 0)  # Default to class 0
            
            x[i] = torch.from_numpy(node_features)
            y[i] = node_label
        
        # Process edges with validation
        edges = list(graph.edges(data='weight'))
        if not edges:
            print("Warning: Graph contains no edges")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, len(edges)*2), dtype=torch.long)
            edge_attr = torch.zeros(len(edges)*2, dtype=torch.float32)
            
            for i, (u, v, w) in enumerate(edges):
                # Handle missing weights
                edge_weight = w if w is not None else 1.0
                edge_index[:, i*2] = torch.tensor([u, v])
                edge_index[:, i*2+1] = torch.tensor([v, u])
                edge_attr[i*2] = edge_attr[i*2+1] = edge_weight
        
        print("PyG Conversion Complete")
        return Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.view(-1, 1).to(device),
            y=y.to(device),
            num_nodes=len(nodes)
        )
    except Exception as e:
        print(f"PyG conversion failed: {str(e)}")
        raise
import torch
from torch_geometric.data import Data
from models.GAT import GAT
from typing import Dict, Any
import numpy as np
import torch.cuda as cuda

class NetworkAttackClassifier:
    def __init__(self, model_path: str, expected_features: int = 13, batch_size: int = 1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Initialize model
        self.model = GAT(num_features=expected_features).to(self.device)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded on {self.device} | Max batch size: {batch_size}")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        


    def classify(self, graph_data: Data) -> Dict[str, Any]:
        """Robust classification with full graph support"""
        # Convert data if needed
        if not isinstance(graph_data.x, torch.Tensor):
            graph_data.x = torch.tensor(graph_data.x, dtype=torch.float32)
        
        # Move everything to device
        graph_data = graph_data.to(self.device)
        num_nodes = graph_data.x.size(0)
        
        # Prepare output tensors
        all_logits = torch.zeros((num_nodes, 2), device='cpu')  # Assuming binary classification
        
        with torch.no_grad():
            # Process all nodes at once if possible, otherwise use batches
            if num_nodes <= self.batch_size or self.device.type == 'cpu':
                # Full graph processing (works on CPU or small graphs)
                logits = self.model(graph_data.x, graph_data.edge_index, 
                                graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None)
                all_logits = logits.cpu()
            else:
                # GPU batch processing with proper edge handling
                edge_index = graph_data.edge_index
                edge_attr = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None
                
                for i in range(0, num_nodes, self.batch_size):
                    batch_nodes = slice(i, min(i + self.batch_size, num_nodes))
                    
                    # Create subgraph for current batch
                    batch_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    batch_mask[batch_nodes] = True
                    
                    # Get edges connecting nodes in this batch
                    edge_mask = batch_mask[edge_index[0]] | batch_mask[edge_index[1]]
                    sub_edge_index = edge_index[:, edge_mask]
                    
                    # Remap node indices for the subgraph
                    node_mapping = torch.zeros(num_nodes, dtype=torch.long)
                    node_mapping[batch_mask] = torch.arange(batch_mask.sum())
                    sub_edge_index = node_mapping[sub_edge_index]
                    
                    # Process batch
                    x_batch = graph_data.x[batch_nodes]
                    sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
                    
                    logits = self.model(x_batch, sub_edge_index, sub_edge_attr)
                    all_logits[batch_nodes] = logits.cpu()
                    
                    # Clean up
                    del x_batch, sub_edge_index, logits
                    torch.cuda.empty_cache()
        
        # Get predictions
        preds = torch.argmax(all_logits, dim=1)
        probs = torch.softmax(all_logits, dim=1)
        
        return {
            'nodes': getattr(graph_data, 'node_names', [f"node_{i}" for i in range(num_nodes)]),
            'predictions': preds.numpy(),
            'probabilities': probs.numpy()
        }

    def visualize_results(self, raw_graph: Dict, results: Dict, output_file: str = "attack_graph.html"):
        """Unchanged visualization code"""
        try:
            import networkx as nx
            from pyvis.network import Network
            
            if not results['nodes']:
                print("Warning: No nodes to visualize")
                return
                
            G = nx.Graph()
            
            # Add nodes with classification results
            for i, node in enumerate(results['nodes']):
                G.add_node(
                    node,
                    label=f"{node}\nStatus: {'ATTACK' if results['predictions'][i] else 'Normal'}",
                    color='red' if results['predictions'][i] else 'green',
                    size=20 if results['predictions'][i] else 10,
                    title=f"Confidence: {results['probabilities'][i][results['predictions'][i]]:.2%}"
                )
            
            # Add edges if they exist
            if 'edge_index' in raw_graph and raw_graph['edge_index'].numel() > 0:
                edge_index = raw_graph['edge_index'].cpu().numpy() if isinstance(
                    raw_graph['edge_index'], torch.Tensor) else raw_graph['edge_index']
                
                for src, dst in zip(edge_index[0], edge_index[1]):
                    if src < len(results['nodes']) and dst < len(results['nodes']):
                        src_node = results['nodes'][src]
                        dst_node = results['nodes'][dst]
                        G.add_edge(src_node, dst_node)
            
            # Generate visualization
            net = Network(height="750px", width="100%", notebook=False)
            net.from_nx(G)
            net.set_options("""{
            "physics": {
                "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
            }""")
            net.save_graph(output_file)
            print(f"Visualization saved to {output_file}")
        
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
import torch
from torch_geometric.data import Data
from models.GAT import GAT
from typing import Dict, Any
import numpy as np

class NetworkAttackClassifier:
    def __init__(self, model_path: str, expected_features: int = 13, batch_size: int = 1024):
        self.model = GAT(num_features=expected_features)
        self.batch_size = batch_size  # Process nodes in batches
        
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def classify(self, graph_data: Data) -> Dict[str, Any]:
        """Classify nodes in the graph"""
        with torch.no_grad():
            logits = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr
            )
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
        return {
            'nodes': getattr(graph_data, 'node_names', 
                           [f"node_{i}" for i in range(graph_data.x.shape[0])]),
            'predictions': preds.cpu().numpy(),
            'probabilities': probs.cpu().numpy()
        }

    def visualize_results(self, raw_graph: Dict, results: Dict, output_file: str = "attack_graph.html"):
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
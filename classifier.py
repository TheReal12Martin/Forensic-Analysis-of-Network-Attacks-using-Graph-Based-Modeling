import torch
from torch_geometric.data import Data
from models.GAT import GAT
from typing import Dict, Any
import numpy as np

class NetworkAttackClassifier:
    def __init__(self, model_path: str, expected_features: int = 13):
        """Initialize classifier with dimension verification"""
        self.model = GAT(num_features=expected_features)
        
        try:
            state_dict = torch.load(model_path)
            
            # Verify feature dimensions
            if 'convs.0.lin.weight' in state_dict:
                loaded_features = state_dict['convs.0.lin.weight'].shape[1]
                if loaded_features != expected_features:
                    print(f"Warning: Model expects {loaded_features} features, "
                          f"but {expected_features} were provided. Attempting adaptation.")
            
            # Load with strict=False to handle dimension mismatches
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
        """Generate interactive visualization"""
        import networkx as nx
        from pyvis.network import Network
        
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
        
        # Add edges from original graph
        edge_index = raw_graph['edge_index'].cpu().numpy() if isinstance(
            raw_graph['edge_index'], torch.Tensor) else raw_graph['edge_index']
        
        for src, dst in zip(edge_index[0], edge_index[1]):
            src_node = results['nodes'][src]
            dst_node = results['nodes'][dst]
            G.add_edge(src_node, dst_node)
        
        # Generate visualization
        net = Network(height="750px", width="100%", notebook=False)
        net.from_nx(G)
        net.set_options("""
        {
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
        }
        """)
        net.show(output_file)
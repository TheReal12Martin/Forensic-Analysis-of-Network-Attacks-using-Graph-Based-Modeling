import torch
from torch_geometric.data import Data
from models.GAT import GAT
from typing import Dict, Any
import numpy as np
import torch.cuda as cuda

class NetworkAttackClassifier:
    def __init__(self, model_path: str, expected_features: int = 13, batch_size: int = 1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GAT(num_features=expected_features).to(self.device)
        self.batch_size = batch_size
        
        # Enable FP16 if supported (1650 Ti has partial FP16 support)
        if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            self.model = self.model.half()
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded on {self.device} | Batch size: {batch_size} | FP16: {next(self.model.parameters()).dtype == torch.float16}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def classify(self, graph_data: Data) -> Dict[str, Any]:
        """Classify nodes in batches with GPU memory checks"""
        if not isinstance(graph_data.x, torch.Tensor):
            graph_data.x = torch.tensor(graph_data.x, 
                                      dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
        
        x_batches = torch.split(graph_data.x, self.batch_size)
        all_logits = []
        
        with torch.no_grad():
            edge_index = graph_data.edge_index.to(self.device)
            edge_attr = graph_data.edge_attr.to(self.device) if hasattr(graph_data, 'edge_attr') else None
            
            for i, batch in enumerate(x_batches):
                batch = batch.to(self.device)
                
                # Memory check for 4GB GPUs
                if self.device.type == 'cuda':
                    allocated = cuda.memory_allocated(0) / 1024**3
                    if allocated > 3.5:  # Leave 0.5GB buffer
                        torch.cuda.empty_cache()
                        raise RuntimeError(f"GPU memory full (allocated: {allocated:.2f}GB). Reduce --batch_size.")
                
                logits = self.model(batch, edge_index, edge_attr)
                all_logits.append(logits.cpu())  # Offload to CPU immediately
                
                del batch  # Free GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            logits = torch.cat(all_logits)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
        return {
            'nodes': getattr(graph_data, 'node_names', 
                           [f"node_{i}" for i in range(graph_data.x.shape[0])]),
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
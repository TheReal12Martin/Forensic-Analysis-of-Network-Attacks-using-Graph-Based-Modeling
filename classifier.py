import torch
from torch_geometric.data import Data
from models.GAT import GAT
from typing import Dict, Any
import numpy as np

class NetworkAttackClassifier:
    def __init__(self, model_path: str, device: torch.device, expected_features: int = 13, batch_size: int = 1024):
        print("\n=== INITIALIZING CLASSIFIER ===")
        print(f"[DEBUG] Model path: {model_path}")
        print(f"[DEBUG] Device: {device}")
        print(f"[DEBUG] Expected features: {expected_features}")
        print(f"[DEBUG] Batch size: {batch_size}")
        
        self.device = device
        self.batch_size = batch_size
        
        # Initialize model with GPU optimizations
        print("[DEBUG] Initializing GAT model...")
        self.model = GAT(num_features=expected_features).to(self.device)
        
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            print("[DEBUG] Enabled TensorFloat-32 for CUDA")
        
        # Load weights with security protection
        try:
            print("[DEBUG] Loading model weights...")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Model loaded on {self.device} | Batch: {batch_size}")
            if self.device.type == 'cuda':
                mem = torch.cuda.memory_allocated()/1024**3
                print(f"[DEBUG] GPU Memory Allocated: {mem:.2f}GB")
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def classify(self, graph_data: Data) -> Dict[str, Any]:
        """Optimized classification with memory management"""
        print("\n=== STARTING CLASSIFICATION ===")
        print(f"[DEBUG] Input graph nodes: {graph_data.num_nodes}")
        print(f"[DEBUG] Input graph edges: {graph_data.edge_index.shape[1]}")
        
        # Ensure data is on correct device
        if not isinstance(graph_data.x, torch.Tensor):
            dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            print(f"[DEBUG] Converting features to {dtype} tensor")
            graph_data.x = torch.tensor(graph_data.x, dtype=dtype).to(self.device)
        
        graph_data = graph_data.to(self.device)
        num_nodes = graph_data.x.size(0)
        print(f"[DEBUG] Processing {num_nodes} nodes")
        
        # Adjust batch size based on available memory
        safe_batch = min(self.batch_size, num_nodes)
        if self.device.type == 'cuda':
            safe_batch = min(safe_batch, 2048)
            print(f"[DEBUG] Adjusted batch size: {safe_batch} (original: {self.batch_size})")
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
            # Full graph inference if possible
            try:
                print("[DEBUG] Attempting full graph inference...")
                logits = self.model(graph_data.x, graph_data.edge_index, 
                                 graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None)
                print("[DEBUG] Full graph inference successful")
            except RuntimeError as e:
                print(f"[DEBUG] Full graph failed, falling back to batches: {str(e)}")
                logits = self._batch_inference(graph_data, safe_batch)
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        print("[DEBUG] Classification complete")
        print(f"[DEBUG] Predictions shape: {preds.shape}")
        print(f"[DEBUG] Probabilities shape: {probs.shape}")
        
        return {
            'nodes': getattr(graph_data, 'node_names', [f"node_{i}" for i in range(num_nodes)]),
            'predictions': preds,
            'probabilities': probs
        }

    def _batch_inference(self, graph_data: Data, batch_size: int) -> torch.Tensor:
        """Process large graphs in batches"""
        print(f"\n=== BATCH PROCESSING ===")
        print(f"[DEBUG] Starting batch inference with size {batch_size}")
        all_logits = []
        edge_index = graph_data.edge_index
        
        for i in range(0, graph_data.num_nodes, batch_size):
            # Print progress every 10%
            if i % max(1, graph_data.num_nodes//10) == 0:
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated()/1024**3
                    print(f"[DEBUG] Processing batch {i}/{graph_data.num_nodes} | GPU Mem: {mem:.2f}GB")
                else:
                    print(f"[DEBUG] Processing batch {i}/{graph_data.num_nodes}")
            
            # Create batch subgraph
            batch_nodes = slice(i, min(i + batch_size, graph_data.num_nodes))
            edge_mask = (edge_index[0] >= i) & (edge_index[0] < i + batch_size)
            
            batch = Data(
                x=graph_data.x[batch_nodes],
                edge_index=edge_index[:, edge_mask],
                edge_attr=graph_data.edge_attr[edge_mask] if hasattr(graph_data, 'edge_attr') else None
            )
            
            # Process batch
            with torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                batch_logits = self.model(batch.x, batch.edge_index, batch.edge_attr)
            
            all_logits.append(batch_logits.cpu())
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print("[DEBUG] Batch processing complete")
        return torch.cat(all_logits).to(self.device)

    def visualize_results(self, raw_graph: Dict, results: Dict, output_file: str = "attack_graph.html"):
        """Visualization with debug info"""
        print("\n=== VISUALIZATION ===")
        print(f"[DEBUG] Output file: {output_file}")
        print(f"[DEBUG] Nodes to visualize: {len(results['nodes'])}")
        
        try:
            import networkx as nx
            from pyvis.network import Network
            
            G = nx.Graph()
            attack_count = sum(results['predictions'])
            print(f"[DEBUG] Creating graph with {attack_count} attack nodes")
            
            for i, node in enumerate(results['nodes']):
                G.add_node(
                    node,
                    label=f"{node}\nStatus: {'ATTACK' if results['predictions'][i] else 'Normal'}",
                    color='red' if results['predictions'][i] else 'green',
                    title=f"Confidence: {results['probabilities'][i][results['predictions'][i]]:.2%}"
                )
            
            if 'edge_index' in raw_graph:
                edge_index = raw_graph['edge_index'].cpu() if isinstance(raw_graph['edge_index'], torch.Tensor) else raw_graph['edge_index']
                print(f"[DEBUG] Adding {edge_index.shape[1]} edges")
                for src, dst in zip(edge_index[0], edge_index[1]):
                    if src < len(results['nodes']) and dst < len(results['nodes']):
                        G.add_edge(results['nodes'][src], results['nodes'][dst])
            
            net = Network(height="750px", width="100%", notebook=False)
            net.from_nx(G)
            net.save_graph(output_file)
            print(f"✅ Visualization saved to {output_file}")
        except Exception as e:
            print(f"❌ Visualization failed: {str(e)}")
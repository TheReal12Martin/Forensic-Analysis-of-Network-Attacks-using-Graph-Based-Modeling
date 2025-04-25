import json
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
        """Memory-optimized batch processing"""
        print(f"\n=== BATCH PROCESSING ===")
        print(f"[DEBUG] Starting batch inference with size {batch_size}")
        
        # Move all data to CPU for preprocessing
        edge_index = graph_data.edge_index.cpu().numpy()
        x_numpy = graph_data.x.cpu().numpy()
        edge_attr_numpy = graph_data.edge_attr.cpu().numpy() if hasattr(graph_data, 'edge_attr') else None
        
        all_logits = []
        
        for i in range(0, graph_data.num_nodes, batch_size):
            batch_end = min(i + batch_size, graph_data.num_nodes)
            
            # Find edges for this batch (CPU operation)
            mask = (edge_index[0] >= i) & (edge_index[0] < batch_end)
            batch_edge_index = edge_index[:, mask] - i  # Adjust indices
            
            # Create batch on CPU first
            batch_x = torch.as_tensor(x_numpy[i:batch_end], dtype=torch.float16)
            batch_edge_index = torch.as_tensor(batch_edge_index, dtype=torch.long)
            
            if edge_attr_numpy is not None:
                batch_edge_attr = torch.as_tensor(edge_attr_numpy[mask], dtype=torch.float16)
            else:
                batch_edge_attr = None
            
            # Move to GPU just before processing
            batch = Data(
                x=batch_x.to(self.device),
                edge_index=batch_edge_index.to(self.device),
                edge_attr=batch_edge_attr.to(self.device) if batch_edge_attr is not None else None
            )
            
            # Process with automatic mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                batch_logits = self.model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Immediately move back to CPU and clean up
            all_logits.append(batch_logits.cpu())
            del batch, batch_logits
            
            # Print progress every 10%
            if (i // batch_size) % max(1, (graph_data.num_nodes // batch_size) // 10) == 0:
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated()/1024**3
                    print(f"[DEBUG] Processed {i}/{graph_data.num_nodes} | GPU Mem: {mem:.2f}GB")
                else:
                    print(f"[DEBUG] Processed {i}/{graph_data.num_nodes}")
            
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



    def save_results(self, raw_graph: Dict, results: Dict, output_file: str = "results.json"):
        """Save classification results to JSON file"""
        data_to_save = {
            'nodes': results['nodes'],
            'predictions': results['predictions'].tolist() if isinstance(results['predictions'], np.ndarray) else results['predictions'],
            'probabilities': results['probabilities'].tolist() if isinstance(results['probabilities'], np.ndarray) else results['probabilities'],
            'edge_index': raw_graph['edge_index'].tolist() if isinstance(raw_graph['edge_index'], torch.Tensor) else raw_graph['edge_index']
        }
        
        with open(output_file, 'w') as f:
            json.dump(data_to_save, f)
        
        print(f"Results saved to {output_file}")


    def save_for_d3(self, raw_graph: Dict, results: Dict, output_file: str = "data/graph.json") -> None:
        """
        Ultra-robust JSON saver that:
        1. Handles NaN/Infinity values
        2. Validates all data types
        3. Uses atomic writes
        """
        import json
        import math
        import numpy as np
        from pathlib import Path

        class SafeJSONEncoder(json.JSONEncoder):
            def encode(self, obj):
                return super().encode(self._clean(obj))

            def _clean(self, obj):
                if isinstance(obj, (float, np.floating)):
                    if math.isnan(obj) or math.isinf(obj):
                        return 0.0  # Replace NaN/Inf with 0
                    return obj
                elif isinstance(obj, (np.generic, np.ndarray)):
                    return self._clean(obj.item() if obj.size == 1 else obj.tolist())
                elif isinstance(obj, dict):
                    return {k: self._clean(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [self._clean(x) for x in obj]
                return obj

        # --- Input Validation ---
        try:
            if not all(k in results for k in ('nodes', 'predictions', 'probabilities')):
                raise ValueError("Missing required keys in results")

            # --- Build Nodes ---
            nodes = []
            for i, (node_id, pred, prob) in enumerate(zip(
                results['nodes'],
                results['predictions'],
                results['probabilities']
            )):
                try:
                    confidence = float(prob[int(pred)])
                    if math.isnan(confidence):
                        confidence = 0.0  # Replace NaN
                    
                    nodes.append({
                        "id": str(node_id),
                        "group": int(bool(pred)),
                        "confidence": confidence
                    })
                except Exception as e:
                    raise ValueError(f"Invalid node data at index {i}: {str(e)}") from e

            # --- Build Links ---
            links = []
            edge_index = raw_graph.get('edge_index', [[], []])
            for src_idx, dst_idx in zip(edge_index[0], edge_index[1]):
                try:
                    if src_idx < len(nodes) and dst_idx < len(nodes):
                        links.append({
                            "source": str(results['nodes'][src_idx]),
                            "target": str(results['nodes'][dst_idx])
                        })
                except Exception as e:
                    raise ValueError(f"Invalid edge data: {str(e)}") from e

            # --- Atomic Write ---
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            temp_file = f"{output_file}.tmp"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {"nodes": nodes, "links": links},
                    f,
                    cls=SafeJSONEncoder,
                    indent=2,
                    ensure_ascii=False
                )
            
            # Validate the file
            with open(temp_file, 'r', encoding='utf-8') as f:
                json.load(f)  # Test parse
            
            Path(temp_file).replace(output_file)
            print(f"✅ Saved validated graph to {output_file}")

        except Exception as e:
            if 'temp_file' in locals():
                Path(temp_file).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save graph: {str(e)}") from e

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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Handle different save formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapped models
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Model loaded on {self.device} | Batch: {batch_size}")
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        self._verify_model_weights()

    def _verify_model_weights(self):
        """New method to check model weights"""
        print("\n=== MODEL WEIGHT VERIFICATION ===")
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                if 'lin' in name and param.dim() == 2:
                    print("Feature weights for attack class:")
                    print(param.data[1].cpu().numpy())

    def classify(self, graph_data: Data) -> Dict[str, Any]:
        """Complete updated classification with dynamic scaling"""
        print("\n=== STARTING CLASSIFICATION ===")
        
        # --- Input Validation (unchanged) ---
        if graph_data is None or not hasattr(graph_data, 'x'):
            return {'nodes': [], 'predictions': [], 'probabilities': []}
            
        # --- Feature Scaling ---
        graph_data.x = graph_data.x * 2.0 - 1.0  # Scale [0,1] -> [-1,1]
        
        # --- Model Inference ---
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
            try:
                logits = self.model(graph_data.x, graph_data.edge_index,
                                  graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None)
                
                # Dynamic temperature scaling
                logit_range = logits.max() - logits.min()
                temperature = max(5.0, logit_range.item() / 2.0)
                scaled_probs = torch.softmax(logits / temperature, dim=1)
                
                print("\n[DEBUG] Model Output:")
                print(f"Temperature: {temperature:.2f}")
                print(f"Logits - Benign: {logits[:,0].mean().item():.2f} ± {logits[:,0].std().item():.2f}")
                print(f"Logits - Attack: {logits[:,1].mean().item():.2f} ± {logits[:,1].std().item():.2f}")
                
            except Exception as e:
                print(f"❌ Inference failed: {str(e)}")
                return {'nodes': [], 'predictions': [], 'probabilities': []}

        # --- Adaptive Thresholding ---
        attack_probs = 1 - scaled_probs[:, 1].cpu().numpy()
        threshold = max(0.6, np.percentile(attack_probs, 99))  # Increase minimum threshold
        
        # --- Final Predictions ---
        preds = (attack_probs > threshold).astype(int)
        attack_indices = np.where(preds == 1)[0]
        adjusted_probs = np.where(
            preds[:, None] == 1,  # Check if prediction is attack
            1 - scaled_probs.cpu().numpy(),  # Invert probabilities for attacks
            scaled_probs.cpu().numpy()       # Keep original for benign
        )
        
        print(f"\n🔴 Detected {len(attack_indices)} potential attacks (threshold={threshold:.2f})")
        print(f"Attack Probability Range: {attack_probs.min():.4f}-{attack_probs.max():.4f}")
        
        # Fallback analysis if no attacks detected
        if len(attack_indices) == 0:
            suspicious = np.argsort(attack_probs)[-5:][::-1]  # Top 5 most suspicious
            print("\n⚠️ Top Suspicious Nodes:")
            for idx in suspicious:
                print(f"{graph_data.nodes[idx]}: {attack_probs[idx]:.2%}")
        
        return {
            'nodes': graph_data.nodes,
            'predictions': preds,
            'probabilities': adjusted_probs
        }
    
    def verify_model(self, graph_data: Data):
        """Verify model responds to synthetic attacks"""
        print("\n=== MODEL VERIFICATION ===")
        
        # Create synthetic attack features
        benign_node = graph_data.x[0].clone()
        attack_node = graph_data.x[0].clone()
        
        # Modify features that should indicate attack
        attack_features = {
            0: 1000,   # High connection count
            4: 1500,   # Large mean packet size
            10: 50,     # Many flags
            11: 20,     # High flag variation
            12: 1.0     # Suspicious flag ratio
        }
        
        for idx, val in attack_features.items():
            attack_node[idx] = val
        
        # Create test data
        test_data = Data(
            x=torch.stack([benign_node, attack_node]),
            edge_index=torch.tensor([[0,1], [1,0]]).t(),
            edge_attr=torch.tensor([1.0, 1.0])
        )
        
        # Run inference
        with torch.no_grad():
            logits = self.model(test_data.x, test_data.edge_index, test_data.edge_attr)
            probs = torch.softmax(logits, dim=1)
        
        print("\n[TEST] Benign Node:")
        print(f"Features: {benign_node.tolist()}")
        print(f"Probabilities: {probs[0].tolist()}")
        
        print("\n[TEST] Attack Node:")
        print(f"Features: {attack_node.tolist()}")
        print(f"Probabilities: {probs[1].tolist()}")
        
        return probs[1][1].item() > 0.5  # Returns True if attack detected

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
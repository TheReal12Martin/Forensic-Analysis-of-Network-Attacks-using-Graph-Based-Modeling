import numpy as np
import torch
from collections import defaultdict

class ModelValidator:
    def __init__(self, model, raw_data, pyg_data, ip_mapping):
        self.model = model
        self.raw_data = raw_data
        self.pyg_data = pyg_data
        self.ip_mapping = ip_mapping
        self.node_to_ip = {v: k for k, v in ip_mapping.items()}  # Reverse mapping

    def check_ip_leakage(self):
        """Check if predictions are overly dependent on IPs"""
        with torch.no_grad():
            preds = self.model(self.pyg_data.x, self.pyg_data.edge_index).argmax(dim=1).cpu().numpy()
        
        # Track prediction consistency per IP
        ip_pred_counts = {}
        for ip, node_id in self.ip_mapping.items():
            if node_id < len(preds):
                pred = preds[node_id]
                if ip not in ip_pred_counts:
                    ip_pred_counts[ip] = {'total': 0, 'pred_0': 0, 'pred_1': 0}
                ip_pred_counts[ip]['total'] += 1
                ip_pred_counts[ip][f'pred_{pred}'] += 1
        
        # Calculate leakage score
        consistent_ips = sum(1 for ip in ip_pred_counts 
                           if ip_pred_counts[ip]['pred_0'] == 0 or 
                           ip_pred_counts[ip]['pred_1'] == 0)
        
        leakage_score = consistent_ips / len(ip_pred_counts) if ip_pred_counts else 0
        
        return {
            'leakage_score': leakage_score,
            'total_ips': len(ip_pred_counts),
            'consistent_ips': consistent_ips
        }

    def check_feature_sensitivity(self):
        """Test if IP-derived features dominate predictions"""
        orig_acc = self._evaluate()
        
        # Ablate IP-related features (last 4 dims assumed to be IP-derived)
        modified_x = self.pyg_data.x.clone()
        modified_x[:, -4:] = 0
        modified_acc = self._evaluate(modified_x)
        
        return {
            'original_accuracy': orig_acc,
            'modified_accuracy': modified_acc,
            'delta': orig_acc - modified_acc
        }
    
    def _evaluate(self, x=None):
        x = x if x is not None else self.pyg_data.x
        with torch.no_grad():
            preds = self.model(x, self.pyg_data.edge_index).argmax(dim=1)
            return (preds[self.pyg_data.test_mask] == self.pyg_data.y[self.pyg_data.test_mask]).float().mean().item()
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import classification_report

class ModelValidator:
    def __init__(self, model, raw_data, pyg_data, ip_mapping):
        self.model = model.cpu()
        self.raw_data = raw_data
        self.pyg_data = pyg_data.cpu()
        self.ip_mapping = ip_mapping
        self.node_to_ip = {v: k for k, v in ip_mapping.items()}

    def check_ip_leakage(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(
                self.pyg_data.x, 
                self.pyg_data.edge_index,
                self.pyg_data.edge_attr
            ).argmax(dim=1).numpy()
        
        ip_pred_counts = defaultdict(lambda: {'total': 0, 'pred_counts': defaultdict(int)})
        
        for ip, node_id in self.ip_mapping.items():
            if node_id < len(preds):
                pred = preds[node_id]
                ip_pred_counts[ip]['total'] += 1
                ip_pred_counts[ip]['pred_counts'][pred] += 1
        
        # Calculate leakage metrics
        consistent_ips = 0
        majority_consistent = 0
        total_ips = len(ip_pred_counts)
        
        for ip in ip_pred_counts:
            preds = ip_pred_counts[ip]['pred_counts']
            if len(preds) == 1:
                consistent_ips += 1
            majority_pred = max(preds.items(), key=lambda x: x[1])[0]
            if preds[majority_pred] / ip_pred_counts[ip]['total'] > 0.9:
                majority_consistent += 1
        
        return {
            'leakage_score': consistent_ips / total_ips if total_ips else 0,
            'majority_score': majority_consistent / total_ips if total_ips else 0,
            'total_ips': total_ips,
            'consistent_ips': consistent_ips
        }

    def check_feature_sensitivity(self, feature_groups):
        results = {}
        base_acc = self._evaluate()
        
        for group_name, feature_indices in feature_groups.items():
            modified_x = self.pyg_data.x.clone()
            modified_x[:, feature_indices] = 0
            modified_acc = self._evaluate(modified_x)
            results[group_name] = {
                'original_accuracy': base_acc,
                'modified_accuracy': modified_acc,
                'delta': base_acc - modified_acc
            }
        
        return results
    
    def evaluate_performance(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(
                self.pyg_data.x,
                self.pyg_data.edge_index,
                self.pyg_data.edge_attr
            ).argmax(dim=1)
            
            y_true = self.pyg_data.y
            
            # Convert to numpy for sklearn
            mask = self.pyg_data.test_mask
            y_true_np = y_true[mask].numpy()
            preds_np = preds[mask].numpy()
            
            report = classification_report(
                y_true_np,
                preds_np,
                output_dict=True,
                zero_division=0
            )
            
            return {
                'test_accuracy': report['accuracy'],
                'classification_report': report
            }
    
    def _evaluate(self, x=None):
        self.model.eval()
        x = x if x is not None else self.pyg_data.x
        
        with torch.no_grad():
            preds = self.model(
                x,
                self.pyg_data.edge_index,
                self.pyg_data.edge_attr
            ).argmax(dim=1)
            
            mask = self.pyg_data.test_mask
            correct = (preds[mask] == self.pyg_data.y[mask]).float().mean()
            return correct.item()
from collections import defaultdict

from sklearn.metrics import classification_report
import torch


class ModelValidator:
    def __init__(self, model, raw_data, pyg_data, node_mapping):
        self.model = model.cpu()
        self.raw_data = raw_data
        self.pyg_data = pyg_data.cpu()
        self.node_mapping = node_mapping
        self.reverse_mapping = {v: k for k, v in node_mapping.items()}

    def check_behavioral_leakage(self):
        """Check for overfitting to protocol patterns"""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.pyg_data.x, self.pyg_data.edge_index).argmax(dim=1).numpy()
        
        # Group predictions by protocol-service combinations
        behavior_groups = defaultdict(list)
        for node_id, pred in enumerate(preds):
            if node_id in self.reverse_mapping:
                behavior = tuple(self.reverse_mapping[node_id].split('_')[:2])  # proto+service
                behavior_groups[behavior].append(pred)
        
        # Calculate consistency
        consistent_groups = sum(1 for group in behavior_groups.values() if len(set(group)) == 1)
        total_groups = len(behavior_groups)
        
        leakage_score = consistent_groups / total_groups if total_groups else 0
        
        return {
            'leakage_score': leakage_score,
            'total_groups': total_groups,
            'consistent_groups': consistent_groups
        }

    def evaluate_performance(self):
        """Fixed performance evaluation"""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.pyg_data.x, self.pyg_data.edge_index).argmax(dim=1)
            y_true = self.pyg_data.y
            
            mask = self.pyg_data.test_mask if hasattr(self.pyg_data, 'test_mask') else torch.ones_like(y_true, dtype=torch.bool)
            
            accuracy = (preds[mask] == y_true[mask]).float().mean().item()
            report = classification_report(
                y_true[mask].numpy(),
                preds[mask].numpy(),
                output_dict=True,
                zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'report': report
            }
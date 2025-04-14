import torch
import numpy as np
from sklearn.metrics import (classification_report, 
                           roc_auc_score, 
                           confusion_matrix, roc_curve, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from utils.feature_importance import analyze_feature_importance
from utils.monitoring import log_resource_usage

def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = preds[data.test_mask]
        y_probs = probs[data.test_mask, 1]  # Malicious class probability
        
        # Enhanced metrics
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)
        
        # Plot curves
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(recall, precision)
        plt.title(f'PR Curve (AP={ap_score:.3f})')
        
        plt.subplot(122)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.plot(fpr, tpr)
        plt.title(f'ROC Curve (AUC={roc_auc_score(y_true, y_probs):.3f})')
        plt.savefig('performance_curves.png')
        
        # Feature importance analysis
        try:
            print("\n🔎 Analyzing feature importance...")
            analyze_feature_importance(model, data)
        except Exception as e:
            print(f"⚠️ Feature importance analysis failed: {str(e)}")
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_probs),
            'ap_score': ap_score,
            'classification_report': classification_report(
                y_true, y_pred, target_names=['Benign', 'Malicious'])
        }
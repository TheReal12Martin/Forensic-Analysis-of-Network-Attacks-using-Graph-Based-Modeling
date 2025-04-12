import torch
import numpy as np
from sklearn.metrics import (classification_report, 
                           roc_auc_score, 
                           confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from utils.monitoring import log_resource_usage

def evaluate_model(model, data, device):
    """Comprehensive model evaluation"""
    try:
        log_resource_usage("Evaluation start")
        model.eval()
        
        with torch.no_grad():
            logits = model(data.x.to(device), data.edge_index.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
            
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = preds[data.test_mask]
            
            # Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            print(f"\n🔍 Test set class distribution: {dict(zip(unique, counts))}")
            
            if len(unique) < 2:
                print("⚠️ Evaluation skipped (only one class)")
                return
            
            # Classification report
            print("\n📊 Classification Report:")
            print(classification_report(
                y_true, y_pred,
                target_names=['Benign', 'Malicious'],
                zero_division=0
            ))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Benign', 'Malicious'],
                       yticklabels=['Benign', 'Malicious'])
            plt.savefig("confusion_matrix.png")
            plt.close()
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_true, probs[data.test_mask])
                print(f"ROC-AUC Score: {roc_auc:.4f}")
            except Exception as e:
                print(f"ROC-AUC calculation failed: {str(e)}")
                
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
    finally:
        log_resource_usage("Evaluation end")
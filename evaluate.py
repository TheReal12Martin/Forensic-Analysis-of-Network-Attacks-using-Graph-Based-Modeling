import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data, device):
    """Evaluates model and prints metrics + plots"""
    model.eval()
    with torch.no_grad():
        # Get predictions
        logits = model(data.x.to(device), data.edge_index.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]  # Malicious class prob
        
        # Ground truth
        y_true = data.y[data.test_mask].cpu().numpy()
        print(y_true)
        y_pred = preds[data.test_mask]
        
        # Skip if only one class exists
        if len(np.unique(y_true)) < 2:
            print("⚠️ Evaluation skipped (only one class in test set)")
            return

        # 1. Classification Report
        print("\n📊 Classification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=['Benign', 'Malicious'],
            labels=[0, 1],
            zero_division=0
        ))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

        # 3. ROC-AUC (if probabilities are available)
        try:
            roc_auc = roc_auc_score(y_true, probs[data.test_mask])
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        except ValueError as e:
            print(f"ROC-AUC failed: {str(e)}")

def save_predictions(data, preds, output_path="predictions.csv"):
    """Saves predictions for analysis"""
    import pandas as pd
    df = pd.DataFrame({
        'true_label': data.y[data.test_mask].cpu().numpy(),
        'predicted': preds[data.test_mask]
    })
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
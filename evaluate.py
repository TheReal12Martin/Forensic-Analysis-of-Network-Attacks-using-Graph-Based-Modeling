import torch
import numpy as np
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    accuracy_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from utils.feature_importance import analyze_feature_importance
from utils.monitoring import log_resource_usage



def _analyze_errors(y_true, y_pred, features, feature_names, partition_idx=None):
    """Perform detailed error analysis"""
    errors = y_true != y_pred
    error_features = features[errors]
    correct_features = features[~errors]
    
    # Feature distribution comparison
    plt.figure(figsize=(15, 6))
    for i, feat in enumerate(feature_names[:5]):  # Top 5 features
        plt.subplot(2, 3, i+1)
        sns.kdeplot(correct_features[:, i], label='Correct')
        sns.kdeplot(error_features[:, i], label='Error')
        plt.title(f'Feature: {feat}')
        plt.legend()
    
    plt.tight_layout()
    filename = f'error_analysis_{partition_idx}.png' if partition_idx else 'error_analysis_final.png'
    plt.savefig(filename)
    plt.close()

def _optimize_threshold(y_true, y_probs):
    """Find optimal decision threshold"""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find threshold meeting recall target
    target_idx = np.argmax(recalls >= Config.TARGET_RECALL)
    optimal_threshold = thresholds[target_idx]
    
    print(f"\n🎯 Threshold Optimization:")
    print(f"- Target Recall: {Config.TARGET_RECALL}")
    print(f"- Optimal Threshold: {optimal_threshold:.3f}")
    print(f"- Achieved Recall: {recalls[target_idx]:.3f}")
    print(f"- Precision at Threshold: {precisions[target_idx]:.3f}")
    
    return optimal_threshold



def evaluate_model(model, data, device, partition_idx=None):
    """Enhanced evaluation with comprehensive metrics"""
    model.eval()
    with torch.no_grad():
        # Get predictions
        logits = model(data.x.to(device), data.edge_index.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = preds[data.test_mask]
        y_probs = probs[data.test_mask, 1]  # Malicious class probability


        feature_names = (
        Config.NUMERIC_FEATURES[:data.x.size(1) - Config.PROTOCOL_DIM] +
        [f'protocol_{i}' for i in range(Config.PROTOCOL_DIM)]
        )
    
        # Error analysis
        _analyze_errors(
            y_true, y_pred, 
            data.x[data.test_mask].cpu().numpy(),
            feature_names,
            partition_idx
        )
        
        # Threshold optimization
        if Config.OPTIMIZE_THRESHOLD:
            optimal_thresh = _optimize_threshold(y_true, y_probs)
            # Apply optimized threshold
            y_pred_opt = (y_probs >= optimal_thresh).astype(int)
            
            print("\n🔍 Optimized Threshold Metrics:")
            print(classification_report(
                y_true, y_pred_opt,
                target_names=['Benign', 'Malicious']
            ))

        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_probs),
            'f1': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'ap_score': average_precision_score(y_true, y_probs),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, 
                target_names=['Benign', 'Malicious'], 
                output_dict=True
            )
        }

        # Generate visualizations
        _generate_metric_plots(y_true, y_probs, metrics, partition_idx)

        # Feature importance analysis (only for final evaluation)
        if partition_idx is None:
            print("\n🔎 Analyzing feature importance...")
            try:
                analyze_feature_importance(model, data)
            except Exception as e:
                print(f"⚠️ Feature importance analysis failed: {str(e)}")
        
        return metrics

def _generate_metric_plots(y_true, y_probs, metrics, partition_idx=None):
    """Generate metric visualization plots"""
    plt.figure(figsize=(15, 5))
    
    # PR Curve
    plt.subplot(131)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision)
    plt.title(f'PR Curve (AP={metrics["ap_score"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # ROC Curve
    plt.subplot(132)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve (AUC={metrics["roc_auc"]:.3f})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
    # Confusion Matrix
    plt.subplot(133)
    sns.heatmap(
        metrics['confusion_matrix'], 
        annot=True, 
        fmt='d',
        cmap='Blues', 
        xticklabels=['Benign', 'Malicious'],
        yticklabels=['Benign', 'Malicious']
    )
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    filename = f'partition_{partition_idx}_metrics.png' if partition_idx else 'final_metrics.png'
    plt.savefig(filename)
    plt.close()
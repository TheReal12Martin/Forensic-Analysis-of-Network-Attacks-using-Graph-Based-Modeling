# utils/feature_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config

def analyze_feature_importance(model, data, n_iter=50):
    """Analyze feature importance with automatic feature count detection"""
    device = next(model.parameters()).device
    num_features = data.x.size(1)  # Automatically detect feature count
    
    # Generate feature names dynamically
    feature_names = (
        Config.NUMERIC_FEATURES[:num_features - Config.PROTOCOL_DIM] +
        [f'protocol_{i}' for i in range(Config.PROTOCOL_DIM)]
    )[:num_features]  # Ensure correct length
    
    original_features = data.x.clone()
    
    # Baseline accuracy
    model.eval()
    with torch.no_grad():
        pred = model(data.x.to(device), data.edge_index.to(device)).argmax(dim=1)
        baseline_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    
    # Permutation importance
    feature_importance = np.zeros(num_features)
    for feat_idx in tqdm(range(num_features), desc="Analyzing features"):
        total_drop = 0
        for _ in range(n_iter):
            shuffled_x = original_features.clone()
            shuffled_x[:, feat_idx] = shuffled_x[torch.randperm(shuffled_x.size(0)), feat_idx]
            
            with torch.no_grad():
                pred = model(shuffled_x.to(device), data.edge_index.to(device)).argmax(dim=1)
                new_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            
            total_drop += (baseline_acc - new_acc)
        
        feature_importance[feat_idx] = total_drop / n_iter
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importance, tick_label=feature_names)
    plt.title("Feature Importance (Accuracy Drop When Shuffled)")
    plt.xlabel("Mean Accuracy Decrease")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    
    return feature_importance
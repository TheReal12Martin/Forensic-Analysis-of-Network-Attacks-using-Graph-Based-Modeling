import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from config import Config

def remove_duplicates(data):
    """Enhanced duplicate removal with feature threshold"""
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(data.x.cpu().numpy())
    
    # Find duplicates above similarity threshold
    duplicates = set()
    for i in range(len(sim_matrix)):
        for j in range(i+1, len(sim_matrix)):
            if sim_matrix[i,j] > 0.99:  # 99% similarity threshold
                duplicates.add(j)
    
    if duplicates:
        print(f"Found {len(duplicates)} near-duplicates (≥99% similarity)")
        keep_mask = torch.ones(len(data.x), dtype=torch.bool)
        keep_mask[list(duplicates)] = False
        
        # Create new data without duplicates
        new_data = Data(
            x=data.x[keep_mask],
            edge_index=torch_geometric.utils.subgraph(
                keep_mask, 
                data.edge_index,
                relabel_nodes=True
            )[0],
            edge_attr=data.edge_attr,
            y=data.y[keep_mask],
            train_mask=data.train_mask[keep_mask],
            val_mask=data.val_mask[keep_mask],
            test_mask=data.test_mask[keep_mask]
        )
        return new_data
    return data

def verify_data_consistency(data):
    """Enhanced consistency checks"""
    issues = []
    
    # Check feature-label alignment
    if data.x.shape[0] != data.y.shape[0]:
        issues.append(f"Feature/label size mismatch: {data.x.shape[0]} vs {data.y.shape[0]}")
    
    # Check for NaN values
    if torch.any(torch.isnan(data.x)):
        issues.append("NaN values detected in features")
    
    # Check for duplicates
    if len(torch.unique(data.x, dim=0)) < len(data.x):
        issues.append(f"Duplicate samples detected: {len(data.x) - len(torch.unique(data.x, dim=0))}")
        data = remove_duplicates(data)
    
    # Check label leakage
    train_nodes = set(torch.where(data.train_mask)[0].tolist())
    val_nodes = set(torch.where(data.val_mask)[0].tolist())
    test_nodes = set(torch.where(data.test_mask)[0].tolist())
    
    if train_nodes & val_nodes:
        issues.append(f"Label leakage: {len(train_nodes & val_nodes)} nodes in both train/val")
    if train_nodes & test_nodes:
        issues.append(f"Label leakage: {len(train_nodes & test_nodes)} nodes in both train/test")
    if val_nodes & test_nodes:
        issues.append(f"Label leakage: {len(val_nodes & test_nodes)} nodes in both val/test")
    
    if issues:
        print("\n⚠️ Data Consistency Issues Found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\n✅ Data consistency checks passed")
    
    return data, not bool(issues)

def run_baseline_comparison(data):
    """Compare against Random Forest baseline"""
    X_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()
    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=Config.RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred)
    
    print(f"\n🔍 Baseline Comparison (Random Forest):")
    print(f"- Test F1 Score: {rf_f1:.4f}")
    print(classification_report(
        y_test, rf_pred,
        target_names=['Benign', 'Malicious']
    ))
    
    return rf_f1
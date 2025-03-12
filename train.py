import matplotlib
matplotlib.use('Agg') 
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from models.GAT import GAT
from utils.data_processing import loadAndProcesData, construct_graph, convert_to_pyg_format
from config import Config
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def train():
    # Load and preprocess data
    features, labels, raw_data, class_weights = loadAndProcesData(Config.CSV_FILE, Config.FEATURES_CSV)
    print("Data has been processed")

    # Scale input features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features = pd.DataFrame(features_scaled, columns=features.columns)

    # Undersample the majority class
    num_malicious = np.sum(labels == 1)
    benign_indices = np.where(labels == 0)[0]
    undersampled_benign_indices = np.random.choice(benign_indices, size=num_malicious, replace=False)
    balanced_indices = np.concatenate([undersampled_benign_indices, np.where(labels == 1)[0]])
    np.random.shuffle(balanced_indices)

    # Create balanced dataset
    balanced_features = features.iloc[balanced_indices]
    balanced_labels = labels[balanced_indices]
    balanced_raw_data = raw_data.iloc[balanced_indices]

    print("Data has been balanced using undersampling")

    # Construct graph
    graph = construct_graph(balanced_features, balanced_labels, balanced_raw_data)
    print("Graph has been created")

    # Convert to PyG format
    data = convert_to_pyg_format(graph, balanced_features, balanced_labels)
    print("Data has been converted")

    # Split data into train, val, and test using stratified sampling
    from sklearn.model_selection import train_test_split
    node_indices = np.arange(data.num_nodes)
    train_indices, test_indices = train_test_split(node_indices, test_size=Config.TEST_RATIO, stratify=balanced_labels)
    train_indices, val_indices = train_test_split(train_indices, test_size=Config.VAL_RATIO / (1 - Config.TEST_RATIO), stratify=balanced_labels[train_indices])

    # Create masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    print("Data has been split into train, val, and test sets")

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_features=data.num_features, hidden_channels=Config.HIDDEN_CHANNELS, num_classes=Config.NUM_CLASSES, heads=Config.HEADS).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Training loop
    train_losses = []
    val_losses = []

    print("Start Training Loop")
    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Use cross-entropy loss for multi-class classification
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()
        train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())

        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), 'gat_network_security_model.pth')
    print("Model saved to 'gat_network_security_model.pth'")

    return model, data

if __name__ == "__main__":
    train()
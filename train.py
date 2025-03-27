from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn.functional as F
from models.GAT import GAT
from utils.data_processing import loadAndProcessData, construct_graph, convert_to_pyg_format
from config import Config
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import pandas as pd

def train():
    try:
        # Load and process data
        print("=== Data Loading ===")
        features, labels, raw_data, class_weights = loadAndProcessData(
            Config.CSV_FILE, 
            Config.FEATURES_CSV
        )
        
        # Scale features
        features = pd.DataFrame(
            StandardScaler().fit_transform(features),
            columns=features.columns
        )

        # Build graph
        print("\n=== Graph Construction ===")
        graph = construct_graph(features, labels, raw_data)
        
        # Convert to PyG format
        print("\n=== PyG Conversion ===")
        data = convert_to_pyg_format(graph)
        
        # Train/val/test split
        node_indices = np.arange(data.num_nodes)
        train_idx, test_idx = train_test_split(
            node_indices,
            test_size=Config.TEST_RATIO,
            stratify=data.y.cpu().numpy(),
            random_state=Config.RANDOM_STATE
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=Config.VAL_RATIO/(1-Config.TEST_RATIO),
            stratify=data.y[train_idx].cpu().numpy(),
            random_state=Config.RANDOM_STATE
        )

        # Create masks
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            heads=Config.HEADS
        ).to(device)
        data = data.to(device)
        
        # Training loop with proper best_acc tracking
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=5e-4
        )
        
        best_acc = 0  # Initialize best accuracy tracker
        train_losses = []
        val_accuracies = []
        
        print("\n=== Starting Training ===")
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(
                out[data.train_mask],
                data.y[data.train_mask],
                weight=class_weights.to(device)
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                val_accuracies.append(val_acc.item())
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), 'best_model.pth')

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{Config.EPOCHS} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Val Acc: {val_acc:.4f}')
        
        # Save training curves
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1,2,2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('training_curves.png')
        
        print(f"\nTraining completed. Best validation accuracy: {best_acc:.4f}")
        return model, data, raw_data

    except Exception as e:
        print(f"\nTraining failed: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    train()
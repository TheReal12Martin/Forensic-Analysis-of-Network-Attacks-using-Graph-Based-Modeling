from sklearn.preprocessing import StandardScaler
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
from time import time

def train():
    try:
        # Setup device and seeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(Config.RANDOM_STATE)
        np.random.seed(Config.RANDOM_STATE)
        
        # Load and process data
        print("=== Data Loading ===")
        features, labels, raw_data, class_weights = loadAndProcessData(
            Config.CSV_FILE, 
            Config.FEATURES_CSV
        )
        
        # Ensure class_weights exists
        if class_weights is None:
            # Fallback: create balanced weights if calculation failed
            unique_labels = np.unique(labels)
            class_weights = torch.ones(len(unique_labels), dtype=torch.float)
            print("Warning: Using default class weights")

        # Move class_weights to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = class_weights.to(device)
        
        # Build graph
        print("\n=== Graph Construction ===")
        graph, ip_mapping = construct_graph(features, labels, raw_data)
        
        # Convert to PyG format
        print("\n=== PyG Conversion ===")
        data = convert_to_pyg_format(graph, device='cpu')
        
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
        
        # Move data to device
        data = data.to(device)
        class_weights = class_weights.to(device)
        
        # Initialize model
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            heads=Config.HEADS,
            dropout=Config.DROPOUT
        ).to(device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        best_acc = 0
        train_losses = []
        val_accuracies = []
        
        print("\n=== Starting Training ===")
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = F.cross_entropy(
                out[data.train_mask],
                data.y[data.train_mask],
                weight=class_weights
            )
            
            # L2 regularization
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                val_accuracies.append(val_acc.item())
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), 'best_model.pth')
            
            scheduler.step(val_acc)
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{Config.EPOCHS} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Val Acc: {val_acc:.4f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
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
        return model, data, raw_data, ip_mapping

    except Exception as e:
        print(f"\nTraining failed: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    train()
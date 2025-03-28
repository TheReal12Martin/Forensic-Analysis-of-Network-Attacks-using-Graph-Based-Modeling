import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from time import time
from models.GAT import GAT
from utils.data_processing import loadAndProcessData, construct_graph, convert_to_pyg_format
from config import Config
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def train():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load and process data
        print("=== Loading Data ===")
        features, labels, raw_data = loadAndProcessData(
            Config.CSV_FILE, 
            Config.FEATURES_CSV
        )

        # Build graph
        print("\n=== Building Graph ===")
        graph, ip_mapping = construct_graph(features, labels, raw_data)
        
        # Create splits at graph level
        node_indices = np.arange(len(graph.nodes()))
        node_labels = np.array([graph.nodes[n]['label'] for n in graph.nodes()])
        
        train_idx, test_idx = train_test_split(
            node_indices,
            test_size=Config.TEST_RATIO,
            stratify=node_labels,
            random_state=Config.RANDOM_STATE
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=Config.VAL_RATIO/(1-Config.TEST_RATIO),
            stratify=node_labels[train_idx],
            random_state=Config.RANDOM_STATE
        )
        
        # Add masks to graph
        graph.train_mask = np.zeros(len(graph.nodes()), dtype=bool)
        graph.val_mask = np.zeros(len(graph.nodes()), dtype=bool)
        graph.test_mask = np.zeros(len(graph.nodes()), dtype=bool)
        graph.train_mask[train_idx] = True
        graph.val_mask[val_idx] = True
        graph.test_mask[test_idx] = True

        # Convert to PyG
        data = convert_to_pyg_format(graph, device=device)
        data.train_mask = torch.tensor(graph.train_mask, dtype=torch.bool).to(device)
        data.val_mask = torch.tensor(graph.val_mask, dtype=torch.bool).to(device)
        data.test_mask = torch.tensor(graph.test_mask, dtype=torch.bool).to(device)

        # Initialize model
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            heads=Config.HEADS,  # Correct parameter name
            dropout=Config.DROPOUT,
            num_layers=Config.GAT_LAYERS
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        # Weighted loss function
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, device=device)
        def loss_fn(output, target):
            return F.cross_entropy(output, target, weight=class_weights)

        # Training loop
        best_val_loss = float('inf')
        no_improve = 0
        
        print("\n=== Starting Training ===")
        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                
                if val_loss < best_val_loss - Config.MIN_DELTA:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(model.state_dict(), 'best_model.pt')
                else:
                    no_improve += 1
            
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            if no_improve >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        model.load_state_dict(torch.load('best_model.pt'))
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
            print(f"\nTest Accuracy: {test_acc:.4f}")
        
        return model, data

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
import numpy as np
from models.GAT import GAT
from utils.data_processing import loadAndProcessData, construct_graph, convert_to_pyg_format
from config import Config

def train():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load data
        print("========= Loading Data =========")
        features, labels, raw_data, class_weights = loadAndProcessData(
            Config.CSV_FILE, 
            Config.FEATURES_CSV
        )
        class_weights = class_weights.to(device)

        # Build graph
        print("\n========= Building Graph =========")
        graph, node_map = construct_graph(features, labels, raw_data)
        data = convert_to_pyg_format(graph, device=device)
        data.edge_index = to_undirected(data.edge_index)

        # Train/val/test split
        node_indices = np.arange(data.num_nodes)
        train_idx, test_idx = train_test_split(
            node_indices,
            test_size=Config.TEST_RATIO,
            stratify=data.y.numpy(),
            random_state=Config.RANDOM_STATE
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=Config.VAL_RATIO/(1-Config.TEST_RATIO),
            stratify=data.y[train_idx].numpy(),
            random_state=Config.RANDOM_STATE
        )

        # Create masks
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        # Data loader
        train_loader = NeighborLoader(
            data,
            num_neighbors=Config.NUM_NEIGHBORS,
            batch_size=Config.BATCH_SIZE,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=Config.NUM_WORKERS
        )

        # Model
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            num_classes=Config.NUM_CLASSES
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        # Training loop
        best_val_acc = 0
        for epoch in range(Config.EPOCHS):
            model.train()
            total_loss = correct = total = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(
                    out[batch.train_mask],
                    batch.y[batch.train_mask],
                    weight=class_weights
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                pred = out.argmax(dim=1)
                correct += (pred[batch.train_mask] == batch.y[batch.train_mask]).sum().item()
                total += batch.train_mask.sum().item()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_acc = (val_out.argmax(dim=1)[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 'best_model.pt')
            
            print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}, '
                  f'Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}')

        return model, data, raw_data, node_map

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()
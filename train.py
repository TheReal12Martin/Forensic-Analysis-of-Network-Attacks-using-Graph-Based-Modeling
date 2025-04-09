import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models.GAT import GAT
from config import Config
import gc
import os

from utils.graph_construction import build_graph_from_partition
from utils.pyg_conversion import convert_to_pyg_memory_safe

def train_partition(partition_file):
    """Train only on valid partitions"""
    try:
        if not os.path.exists(partition_file):
            return None, None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract partition number from filename
        partition_num = int(os.path.basename(partition_file).split('_')[1].split('.')[0])
        print(f"\n=== Training on Partition {partition_num} ===")

        # Graph construction with validation
        graph, ip_to_idx = build_graph_from_partition(partition_file)
        if graph is None:
            return None, None

        # PyG conversion
        data = convert_to_pyg_memory_safe(graph, device)
        
        # Verify test set balance
        test_labels = data.y[data.test_mask].cpu().numpy()
        if len(np.unique(test_labels)) < 2:
            print(f"⚠️ Skipping training - test set has only one class")
            return None, None

        # Model setup
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            heads=Config.HEADS,
            num_layers=Config.GAT_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)

        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=Config.LEARNING_RATE,
                                    weight_decay=Config.WEIGHT_DECAY)
        best_val_acc = 0
        best_model_path = f"best_model_partition_{partition_num}.pt"

        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], 
                                 data.y[data.train_mask],
                                 weight=torch.tensor(Config.CLASS_WEIGHTS).to(device))
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_model_path)

        # Early stopping check
        if best_val_acc < 0.6:  # Higher threshold for confidence
            print(f"⚠️ Training failed (val_acc={best_val_acc:.2f} < 0.6)")
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            return None, None

        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)  # Clean up
        
        return model, data

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        # Clean up model file if exists
        if 'best_model_path' in locals() and os.path.exists(best_model_path):
            os.remove(best_model_path)
        return None, None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
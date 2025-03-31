import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models.GAT import GAT
from config import Config
import gc

from utils.graph_construction import build_graph_from_partition
from utils.pyg_conversion import convert_to_pyg_memory_safe

def train_partition(partition_file):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Graph Construction (check if successful)
        graph, ip_to_idx = build_graph_from_partition(partition_file)
        if graph is None or len(graph.nodes) == 0:
            raise ValueError("Graph construction failed - empty or invalid partition")

        # 2. Convert to PyG (validate data)
        data = convert_to_pyg_memory_safe(graph, device)
        if not hasattr(data, 'train_mask'):
            raise ValueError("PyG Data missing train_mask!")

        # 3. Train GAT (return model or raise error)
        model = GAT(
            num_features=data.num_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            heads=Config.HEADS,
            num_layers=Config.GAT_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        best_val_acc = 0.0

        for epoch in range(Config.EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model.state_dict()

        if best_val_acc < 0.5:  # Sanity check
            raise ValueError(f"Training failed (val_acc={best_val_acc:.2f})")
        
        model.load_state_dict(best_model)
        return model, data

    except Exception as e:
        print(f"\n❌ Training crashed on {partition_file}: {str(e)}\n")
        return None, None  # Explicitly return None for error handling
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import Config

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels=128, heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # ===== NEW ARCHITECTURE COMPONENTS =====
        # Input normalization
        self.input_norm = nn.BatchNorm1d(num_features)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=heads,
                dropout=dropout
            ))
        
        # Output layer with controlled scale
        self.convs.append(GATConv(
            hidden_channels * heads,
            2,  # num_classes
            heads=1,  # Single head for output
            concat=False,
            dropout=dropout
        ))
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.5))

        self.temperature_min = 0.1
        self.temperature_max = 2.0
        # ===== END NEW ARCHITECTURE =====

    def forward(self, x, edge_index, edge_attr=None):
        # ===== NEW FORWARD PASS =====
        # Normalize input features
        x = self.input_norm(x)
        
        # Forward pass through GAT layers
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer with controlled output scale
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        clamped_temp = torch.clamp(self.temperature, self.temperature_min, self.temperature_max)
        x = x / (clamped_temp + 1e-8)
        
        return x  # Return raw logits
    
    def visualize_attention(self, edge_index, node_idx=None):
        """Visualize attention patterns"""
        _, att_weights = self.conv1(
            self.x, 
            edge_index, 
            return_attention_weights=True
        )
        
        plt.figure(figsize=(12, 6))
        
        # Histogram of all attention weights
        plt.subplot(121)
        plt.hist(att_weights[1].cpu().numpy(), bins=50)
        plt.title('Attention Weights Distribution')
        
        # Example node's attention
        if node_idx is not None:
            plt.subplot(122)
            node_att = att_weights[1][att_weights[0][0] == node_idx]
            plt.bar(range(len(node_att)), node_att.cpu().numpy())
            plt.title(f'Attention for Node {node_idx}')
        
        plt.tight_layout()
        plt.savefig('attention_visualization.png')
        plt.close()
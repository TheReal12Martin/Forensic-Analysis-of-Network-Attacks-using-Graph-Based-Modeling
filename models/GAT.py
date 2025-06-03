from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels=128, heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
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
        

    def forward(self, x, edge_index, edge_attr=None):
        
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
    
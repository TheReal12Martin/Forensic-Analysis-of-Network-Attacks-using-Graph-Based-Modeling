import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import Config

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels=256, heads=4, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        # Residual connections
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.skip.append(nn.Linear(num_features, hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * heads, 
                hidden_channels, 
                heads=heads, 
                dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            self.skip.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))
        
        # Output layer
        self.convs.append(GATConv(
            hidden_channels * heads, 
            2,  # num_classes
            heads=6, 
            concat=False,
            dropout=dropout
        ))
        
        # Edge importance weighting
        self.edge_encoder = nn.Linear(1, heads)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Edge attention boosting
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr.view(-1, 1))
        
        # Forward pass with residuals
        for i, conv in enumerate(self.convs[:-1]):
            x_res = self.skip[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_attr=edge_attr) + x_res
            x = self.bns[i](x)
            x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        
        return F.log_softmax(x, dim=1)
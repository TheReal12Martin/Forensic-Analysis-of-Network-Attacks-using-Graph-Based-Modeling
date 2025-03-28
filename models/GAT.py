import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4, dropout=0.6, num_layers=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # First GAT layer
        self.conv1 = GATConv(
            in_channels=num_features,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout
        )
        
        # Hidden GAT layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(
                GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout
                )
            )
        
        # Output layer
        self.conv_out = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=num_classes,
            heads=1,  # Single head for final layer
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        for layer in self.hidden_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=3, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=2, dropout=dropout)
        self.conv3 = GATConv(hidden_channels*2, hidden_channels, heads=1, dropout=dropout)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels*heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels*2)
        
        self.skip = nn.Linear(num_features, hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x_init = x.clone()
        
        # First GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        # Second GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        # Third GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Skip connection
        x_skip = self.skip(x_init)
        x = x + x_skip
        
        # Final processing
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import Config

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GATConv(
            num_features,
            hidden_channels,
            heads=Config.HEADS,
            dropout=Config.DROPOUT
        )
        self.conv2 = GATConv(
            hidden_channels * Config.HEADS,
            hidden_channels,
            heads=1,
            dropout=Config.DROPOUT
        )
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = Config.DROPOUT

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import Config

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels=Config.HIDDEN_CHANNELS, 
                 heads=Config.HEADS, num_layers=Config.GAT_LAYERS, 
                 dropout=Config.DROPOUT, num_classes=Config.NUM_CLASSES):
        super(GAT, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, num_classes, 
                                heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)
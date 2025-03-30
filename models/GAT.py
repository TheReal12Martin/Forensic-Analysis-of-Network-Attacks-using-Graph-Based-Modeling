import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import BatchNorm1d
from config import Config

class GAT(nn.Module):
    def __init__(self, num_features=None, hidden_channels=None, num_classes=None,
                 heads=None, dropout=None, num_layers=None):
        super(GAT, self).__init__()
        
        # Use config values if not specified
        self.num_features = num_features if num_features is not None else Config.INPUT_FEATURES
        self.hidden_channels = hidden_channels if hidden_channels is not None else Config.HIDDEN_CHANNELS
        self.num_classes = num_classes if num_classes is not None else Config.NUM_CLASSES
        self.heads = heads if heads is not None else Config.HEADS
        self.dropout = dropout if dropout is not None else Config.DROPOUT
        self.num_layers = num_layers if num_layers is not None else Config.GAT_LAYERS
        
        print("\nGAT Model Configuration:")
        print(f"Input Features: {self.num_features}")
        print(f"Hidden Channels: {self.hidden_channels}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Attention Heads: {self.heads}")
        print(f"Number of Layers: {self.num_layers}")
        
        # Input normalization
        self.bn_input = BatchNorm1d(self.num_features)
        
        # Input projection
        self.input_proj = nn.Linear(self.num_features, self.hidden_channels)
        self.bn_proj = BatchNorm1d(self.hidden_channels)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_channels = self.hidden_channels * self.heads if i > 0 else self.hidden_channels
            self.convs.append(
                GATConv(in_channels, self.hidden_channels, 
                       heads=self.heads, dropout=self.dropout)
            )
            self.bns.append(BatchNorm1d(self.hidden_channels * self.heads))
        
        # Output layer
        self.conv_out = GATConv(
            self.hidden_channels * self.heads,
            self.num_classes,
            heads=1,
            concat=False,
            dropout=self.dropout
        )

        # Class weights
        self.class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)
        
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_out.reset_parameters()

    def forward(self, x, edge_index):
        # Input validation
        if x.shape[1] != self.num_features:
            raise ValueError(f"Input feature dimension mismatch! Expected {self.num_features}, got {x.shape[1]}")

        # Input normalization
        x = self.bn_input(x)
        
        # Project input
        x = F.elu(self.bn_proj(self.input_proj(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.conv_out(x, edge_index)
        return F.log_softmax(x, dim=1)
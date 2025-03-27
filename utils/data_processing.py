import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
import torch
from collections import Counter
import warnings

def loadAndProcessData(data_csv, features_csv):
    """Load data with strict ISO-8859-1 encoding and robust validation"""
    try:
        # Load feature names with ISO-8859-1
        feature_names = pd.read_csv(
            features_csv, 
            encoding='ISO-8859-1',
            on_bad_lines='warn'
        )['Name'].tolist()

        # Load main data with ISO-8859-1
        raw_data = pd.read_csv(
            data_csv,
            header=None,
            encoding='ISO-8859-1',
            dtype=str,  # Read everything as string initially
            low_memory=False,
            on_bad_lines='warn'
        )
        
        # Validate column count
        if len(raw_data.columns) != len(feature_names):
            raise ValueError(
                f"Column mismatch. Expected {len(feature_names)}, got {len(raw_data.columns)}"
            )
        raw_data.columns = feature_names

        # Convert numeric columns
        numeric_cols = [col for col in raw_data.columns 
                       if col not in ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'attack_cat', 'Label']]
        
        for col in numeric_cols:
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
        
        # Clean remaining data
        raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        raw_data.fillna(0, inplace=True)

        # Process labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(raw_data['Label'])
        
        # Compute class weights
        class_weights = torch.tensor(
            compute_class_weight('balanced', classes=np.unique(labels), y=labels),
            dtype=torch.float
        )

        return raw_data[numeric_cols], labels, raw_data, class_weights

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise

def construct_graph(features, labels, raw_data):
    G = nx.Graph()
    ip_to_node = {}  # This will store our IP mapping
    
    for idx, row in raw_data.iterrows():
        src_ip = str(row['srcip'])
        dst_ip = str(row['dstip'])
        
        # Add nodes
        for ip in [src_ip, dst_ip]:
            if ip not in ip_to_node:
                node_id = len(ip_to_node)
                ip_to_node[ip] = node_id
                G.add_node(node_id, 
                          features=features.iloc[idx].values,
                          label=labels[idx])
        
        # Add edge
        G.add_edge(ip_to_node[src_ip], ip_to_node[dst_ip])
    
    print(f"Graph constructed with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G, ip_to_node  # Now returning both graph and mapping

def convert_to_pyg_format(graph, device='cpu'):
    """Convert with ISO-8859-1 compatible features"""
    try:
        x = torch.tensor(
            np.array([graph.nodes[n]['features'] for n in graph.nodes]),
            dtype=torch.float32
        ).to(device)
        
        y = torch.tensor(
            [graph.nodes[n]['label'] for n in graph.nodes],
            dtype=torch.long
        ).to(device)
        
        edge_index = torch.tensor(
            list(graph.edges),
            dtype=torch.long
        ).t().contiguous().to(device)
        
        return Data(x=x, edge_index=edge_index, y=y)
    except Exception as e:
        print(f"PyG conversion failed: {str(e)}")
        raise
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
import torch
from collections import defaultdict
import hashlib
import gc

def loadAndProcessData(data_csv, features_csv):
    """Load data and ensure proper class weights are returned"""
    try:
        # Load feature names
        feature_defs = pd.read_csv(features_csv, encoding='ISO-8859-1')
        if 'Name' not in feature_defs.columns:
            feature_defs.columns = ['Name']  # Assume first column is names

        # Load data
        raw_data = pd.read_csv(data_csv, header=None, names=feature_defs['Name'],
                             encoding='ISO-8859-1', low_memory=False)

        # Process IPs
        for ip_col in ['srcip', 'dstip']:
            if ip_col in raw_data.columns:
                raw_data[ip_col] = raw_data[ip_col].apply(
                    lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 10000
                ).astype('int32')

        # Process labels and ensure class weights
        if 'Label' not in raw_data.columns:
            raise ValueError("Label column not found")
        
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(raw_data['Label'])
        
        # Calculate class weights (critical fix)
        classes, counts = np.unique(labels, return_counts=True)
        class_weights = torch.tensor(
            [1.0 / counts[i] for i in range(len(classes))],  # Inverse frequency weighting
            dtype=torch.float
        )
        
        # Prepare features
        features = raw_data.drop(columns=['Label', 'attack_cat', 'Stime', 'Ltime'], errors='ignore')
        numeric_cols = features.select_dtypes(include=['number']).columns
        features[numeric_cols] = StandardScaler().fit_transform(features[numeric_cols])

        return features, labels, raw_data, class_weights

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise

def construct_graph(features, labels, raw_data, max_nodes=50000):
    """Simplified graph construction"""
    try:
        G = nx.Graph()
        ip_to_node = {}
        
        # Process only first max_nodes unique IPs
        unique_ips = pd.concat([raw_data['srcip'], raw_data['dstip']]).drop_duplicates()[:max_nodes]
        
        for ip in unique_ips:
            ip_to_node[ip] = len(ip_to_node)
            G.add_node(ip_to_node[ip], features=None, label=None)
        
        # Add edges
        for _, row in raw_data.iterrows():
            src_ip = row['srcip']
            dst_ip = row['dstip']
            
            if src_ip in ip_to_node and dst_ip in ip_to_node:
                src_node = ip_to_node[src_ip]
                dst_node = ip_to_node[dst_ip]
                
                if G.has_edge(src_node, dst_node):
                    G[src_node][dst_node]['weight'] += 1
                else:
                    G.add_edge(src_node, dst_node, weight=1)
        
        print(f"Graph constructed with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G, ip_to_node

    except Exception as e:
        print(f"Graph construction failed: {str(e)}")
        raise

def convert_to_pyg_format(graph, device='cpu'):
    """Simplified PyG conversion"""
    try:
        # Create node features matrix
        num_nodes = len(graph.nodes)
        if num_nodes == 0:
            raise ValueError("Empty graph")
            
        # Dummy features if none exist
        if not graph.nodes[0].get('features'):
            feature_size = 10  # Default feature size
            x = torch.randn(num_nodes, feature_size, dtype=torch.float32)
        else:
            x = torch.tensor([graph.nodes[n].get('features', [0]*10) for n in graph.nodes], 
                            dtype=torch.float32)
        
        # Create edge index and attributes
        edge_index = []
        edge_attr = []
        for u, v, data in graph.edges(data=True):
            edge_index.append([u, v])
            edge_attr.append(data.get('weight', 1.0))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        
        # Create labels if they exist
        if graph.nodes[0].get('label') is not None:
            y = torch.tensor([graph.nodes[n].get('label', 0) for n in graph.nodes], 
                            dtype=torch.long)
        else:
            y = torch.zeros(num_nodes, dtype=torch.long)
        
        return Data(x=x.to(device), 
                  edge_index=edge_index.to(device), 
                  edge_attr=edge_attr.to(device), 
                  y=y.to(device))
    
    except Exception as e:
        print(f"PyG conversion failed: {str(e)}")
        raise
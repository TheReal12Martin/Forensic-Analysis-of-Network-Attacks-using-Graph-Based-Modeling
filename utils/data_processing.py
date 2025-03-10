import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
import torch

def loadAndProcesData(csv_file):
    # Load the specific CSV file
    raw_data = pd.read_csv(csv_file)
    
    # Replace invalid values (e.g., '-', 'NaN', 'inf', '-inf') with NaN
    raw_data.replace(['-', 'NaN', 'inf', '-inf'], np.nan, inplace=True)
    
    # Convert numeric columns to numeric, skipping non-numeric columns
    for col in raw_data.columns:
        try:
            raw_data[col] = pd.to_numeric(raw_data[col])  # Convert to numeric
        except (ValueError, TypeError):
            # Skip non-numeric columns
            continue
    
    # Replace infinite values with NaN
    raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN values with 0 or the mean/median of the column
    raw_data.fillna(0, inplace=True)  # Replace with 0 or use raw_data.fillna(raw_data.mean(), inplace=True)
    
    # Exclude non-numeric columns (e.g., IP addresses) from features
    non_numeric_columns = ['Flow ID', ' Source IP', ' Destination IP', ' Label', ' Timestamp']  # Add other non-numeric columns if needed
    features = raw_data.drop(columns=non_numeric_columns)
    
    # Check for infinite or extremely large values in features
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Input X contains infinity or a value too large for dtype('float64')")
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_data[' Label'])  # Use the 'Label' column for labels
    
    return features, labels, raw_data  # Return the raw data for edge construction

def constructGraph(features,labels, raw_data):
    G = nx.Graph()
    
    # Create a mapping from IP addresses to node indices
    ip_to_node = {}
    node_counter = 0
    
    # Add nodes and edges based on communication
    for i, row in raw_data.iterrows():
        src_ip = row[' Source IP']  # Replace 'Source IP' with the actual column name
        dst_ip = row[' Destination IP']  # Replace 'Destination IP' with the actual column name
        
        # Add source IP as a node if not already added
        if src_ip not in ip_to_node:
            ip_to_node[src_ip] = node_counter
            G.add_node(node_counter, features=features[i], label=labels[i])
            node_counter += 1
        
        # Add destination IP as a node if not already added
        if dst_ip not in ip_to_node:
            ip_to_node[dst_ip] = node_counter
            G.add_node(node_counter, features=features[i], label=labels[i])
            node_counter += 1
        
        # Add an edge between source and destination IPs
        src_node = ip_to_node[src_ip]
        dst_node = ip_to_node[dst_ip]
        G.add_edge(src_node, dst_node)
    
    return G

def convertToPygFormat(graph, features, labels):
    # Extract edge indices from the graph
    edge_index = np.array(list(graph.edges), dtype=np.int64).T  # Shape: [2, num_edges]
    
    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float)  # Node features
    y = torch.tensor(labels, dtype=torch.long)     # Node labels
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # Edge indices
    
    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data
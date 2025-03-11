import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
import torch
from collections import Counter

def loadAndProcesData(data_csv, features_csv):
    # Load the feature names with the correct encoding
    try:
        feature_names = pd.read_csv(features_csv, encoding='ISO-8859-1')['Name'].tolist()
    except UnicodeDecodeError:
        # Try other common encodings if the first one fails
        feature_names = pd.read_csv(features_csv, encoding='latin1')['Name'].tolist()
    
    # Load the dataset with the correct encoding
    try:
        raw_data = pd.read_csv(data_csv, header=None, encoding='ISO-8859-1', low_memory=False)
    except UnicodeDecodeError:
        # Try other common encodings if the first one fails
        raw_data = pd.read_csv(data_csv, header=None, encoding='latin1', low_memory=False)
    
    # Assign feature names to the dataset columns
    if len(raw_data.columns) == len(feature_names):
        raw_data.columns = feature_names
    else:
        raise ValueError("Number of columns in the dataset does not match the number of feature names.")
    
    # Replace invalid values (e.g., '-', 'NaN', 'inf', '-inf') with NaN
    raw_data.replace(['-', 'NaN', 'inf', '-inf'], np.nan, inplace=True)
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in raw_data.columns:
        raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')  # Use 'coerce' to convert non-numeric values to NaN
    
    # Replace infinite values with NaN
    raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN values with 0 or the mean/median of the column
    raw_data.fillna(0, inplace=True)  # Replace with 0 or use raw_data.fillna(raw_data.mean(), inplace=True)
    
    # Exclude non-numeric columns (e.g., IP addresses, labels) from features
    non_numeric_columns = ['srcip', 'sport', 'dstip', 'dsport', 'attack_cat', 'Label']  # Adjust based on the dataset
    features = raw_data.drop(columns=non_numeric_columns)
    
    # Ensure all features are numeric
    if not np.issubdtype(features.to_numpy().dtype, np.number):
        raise ValueError("Features contain non-numeric data. Please check the dataset.")
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_data['Label'])  # Use the 'Label' column for labels

    print("Class distribution in the dataset:", Counter(labels))
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    return features, labels, raw_data, class_weights  # Return the raw data and class weights

def construct_graph(features, labels, raw_data):
    G = nx.Graph()
    
    # Create a mapping from IP addresses to node indices
    ip_to_node = {}
    node_counter = 0
    
    # Add nodes and edges based on communication
    for i, row in raw_data.iterrows():
        src_ip = row['srcip']  # Source IP
        dst_ip = row['dstip']  # Destination IP
        
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

def convert_to_pyg_format(graph, features, labels):
    # Extract edge indices from the graph
    edge_index = np.array(list(graph.edges), dtype=np.int64).T  # Shape: [2, num_edges]
    
    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float)  # Node features
    y = torch.tensor(labels, dtype=torch.long)     # Node labels
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # Edge indices
    
    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data
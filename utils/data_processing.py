import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch
from config import Config
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def loadAndProcessData(data_csv, features_csv):
    try:
        print("\n=== Loading Data ===")
        
        # Load feature definitions
        feature_defs = pd.read_csv(features_csv, encoding='ISO-8859-1')
        
        # Generate column names
        column_names = []
        name_counts = {}
        for i in range(len(feature_defs)):
            base_name = feature_defs.iloc[i]['Name']
            if base_name in name_counts:
                name_counts[base_name] += 1
                column_names.append(f"{base_name}_dup{name_counts[base_name]}")
            else:
                name_counts[base_name] = 0
                column_names.append(base_name)
        
        # Load data with generated headers
        raw_data = pd.read_csv(
            data_csv,
            header=None,
            names=column_names,
            low_memory=False
        )
        
        # Process labels
        if Config.LABEL_COLUMN not in raw_data.columns:
            raise ValueError(f"Label column '{Config.LABEL_COLUMN}' not found")
        
        def map_label(label):
            label_str = str(label).lower()
            for pattern, value in Config.LABEL_MAPPING.items():
                if pattern in label_str:
                    return value
            return 1  # Default to malicious
        
        raw_data['Label'] = raw_data[Config.LABEL_COLUMN].apply(map_label)
        
        # Sample data if needed
        if len(raw_data) > Config.SAMPLE_SIZE:
            raw_data = raw_data.sample(
                Config.SAMPLE_SIZE, 
                random_state=Config.RANDOM_STATE
            ).reset_index(drop=True)
        
        # Process features
        numeric_cols = [col for col in raw_data.columns 
                       if col in Config.NUMERIC_FEATURES]
        numeric_features = raw_data[numeric_cols].fillna(0)
        
        categorical_cols = [col for col in raw_data.columns 
                          if col in Config.CATEGORICAL_FEATURES]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        categorical_features = encoder.fit_transform(raw_data[categorical_cols])
        
        # Combine features
        features = np.concatenate([numeric_features.values, categorical_features], axis=1)
        
        # Dimensionality reduction
        if features.shape[1] > Config.MAX_FEATURES:
            features = PCA(n_components=Config.MAX_FEATURES).fit_transform(features)
        
        # Normalization
        features = StandardScaler().fit_transform(features)
        
        return features, raw_data['Label'].values, raw_data

    except Exception as e:
        print(f"\nData loading failed: {str(e)}")
        raise

def construct_graph(features, labels, raw_data):
    try:
        print("\n=== Building Graph ===")
        G = nx.Graph()
        ip_mapping = {}
        
        # Reset indices for alignment
        raw_data = raw_data.reset_index(drop=True)
        
        # First pass: Create nodes and track flows per IP
        ip_flows = defaultdict(list)
        for idx, row in raw_data.iterrows():
            src_ip = row['srcip']
            dst_ip = row['dstip']
            ip_flows[src_ip].append(idx)
            ip_flows[dst_ip].append(idx)
        
        # Create nodes with aggregated features
        all_ips = list(ip_flows.keys())
        for ip in all_ips:
            ip_mapping[ip] = len(ip_mapping)
            flow_indices = ip_flows[ip]
            
            # Aggregate features from all flows
            avg_features = np.mean(features[flow_indices], axis=0)
            # Label as malicious if any flow is malicious
            ip_label = max(labels[flow_indices])
            
            G.add_node(
                ip_mapping[ip],
                features=avg_features,
                label=ip_label,
                ip=ip,
                flow_count=len(flow_indices)
            )
        
        # Second pass: Create edges
        edge_weights = defaultdict(int)
        for _, row in raw_data.iterrows():
            src_idx = ip_mapping[row['srcip']]
            dst_idx = ip_mapping[row['dstip']]
            edge_key = tuple(sorted((src_idx, dst_idx)))
            edge_weights[edge_key] += 1
        
        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)
        
        # Filter edges
        if Config.MIN_EDGE_WEIGHT > 0:
            edges_to_remove = [
                (u, v) for u, v, d in G.edges(data=True) 
                if d['weight'] < Config.MIN_EDGE_WEIGHT
            ]
            G.remove_edges_from(edges_to_remove)
            print(f"Removed {len(edges_to_remove)} weak edges")
        
        print("\n=== Graph Statistics ===")
        print(f"Nodes: {len(G.nodes)}")
        print(f"Edges: {len(G.edges)}")
        print("Class distribution:")
        print(pd.Series([G.nodes[n]['label'] for n in G.nodes]).value_counts())
        
        return G, ip_mapping

    except Exception as e:
        print(f"\nGraph construction failed: {str(e)}")
        raise

def convert_to_pyg_format(graph, device='cpu'):
    try:
        print("\n=== Converting to PyG ===")
        # Get nodes in consistent order
        nodes = sorted(graph.nodes())
        features = np.array([graph.nodes[n]['features'] for n in nodes])
        y = torch.tensor([graph.nodes[n]['label'] for n in nodes], dtype=torch.long)
        
        # Convert edges
        edges = list(graph.edges(data='weight', default=1.0))
        edge_index = torch.tensor([(u, v) for u, v, _ in edges], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([w for _, _, w in edges], dtype=torch.float).view(-1, 1)
        
        # Make undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        print("\n=== PyG Data Summary ===")
        print(f"Node features shape: {features.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge weights shape: {edge_attr.shape}")
        print(f"Classes present: {torch.unique(y).tolist()}")
        
        return Data(
            x=torch.FloatTensor(features).to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            y=y.to(device),
            num_nodes=len(nodes)
        )
    except Exception as e:
        print(f"\nPyG conversion failed: {str(e)}")
        raise
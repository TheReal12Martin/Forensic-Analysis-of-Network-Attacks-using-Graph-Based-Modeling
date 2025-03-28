import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from torch_geometric.data import Data
import torch
from collections import defaultdict
import hashlib
from config import Config
import gc

def loadAndProcessData(data_csv, features_csv):
    try:
        print("\n=== Loading and Processing Data ===")
        
        # Load feature definitions
        feature_defs = pd.read_csv(features_csv, encoding='ISO-8859-1')
        if 'Name' not in feature_defs.columns:
            feature_defs.columns = ['Name'] + feature_defs.columns[1:].tolist()

        # Load data in chunks
        chunks = []
        for chunk in pd.read_csv(data_csv, header=None, chunksize=100000, low_memory=False):
            if len(chunk.columns) == len(feature_defs):
                chunk.columns = feature_defs['Name']
            else:
                chunk.columns = [f'col_{i}' for i in range(len(chunk.columns)-1)] + ['Label']
            chunks.append(chunk)
        raw_data = pd.concat(chunks, ignore_index=True)

        # IP Anonymization
        if Config.IP_ANONYMIZE:
            print("Applying IP anonymization...")
            for ip_col in ['srcip', 'dstip']:
                if ip_col in raw_data.columns:
                    raw_data[ip_col] = raw_data[ip_col].astype(str).apply(
                        lambda x: hashlib.sha256(x.encode()).hexdigest()[:Config.IP_HASH_LENGTH]
                    )

        # Class Balancing
        print("\n=== Class Distribution Before Balancing ===")
        class_counts = raw_data['Label'].value_counts()
        print(class_counts)

        min_samples = min(Config.MIN_SAMPLES_PER_CLASS, class_counts.min())
        balanced_data = pd.concat([
            raw_data[raw_data['Label'] == cls].sample(min_samples, random_state=Config.RANDOM_STATE)
            for cls in class_counts.index
        ])

        print("\n=== Class Distribution After Balancing ===")
        print(balanced_data['Label'].value_counts())

        # Feature Engineering
        print("\n=== Feature Engineering ===")
        features = balanced_data.drop(['Label', 'attack_cat'], axis=1, errors='ignore')
        
        # Create robust features
        with np.errstate(divide='ignore', invalid='ignore'):
            features['flow_ratio'] = np.log1p(features['sbytes']) / np.log1p(features['dbytes'].replace(0, 1))
            features['response_ratio'] = features['Dpkts'] / features['Spkts'].replace(0, 1)
        features = features.fillna(0)

        # Convert categoricals
        for col in ['proto', 'state', 'service']:
            if col in features.columns:
                features[col] = pd.Categorical(features[col]).codes.astype('int8')

        # Feature Normalization
        print("\n=== Feature Normalization ===")
        numeric_cols = features.select_dtypes(include=['number']).columns
        features[numeric_cols] = StandardScaler().fit_transform(features[numeric_cols])
        features[numeric_cols] = np.clip(features[numeric_cols], -5, 5)

        # Class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(balanced_data['Label']),
            y=balanced_data['Label']
        )
        print("\n=== Class Weights ===")
        print(f"Class 0: {class_weights[0]:.4f}, Class 1: {class_weights[1]:.4f}")

        gc.collect()
        return features, balanced_data['Label'].values, balanced_data, torch.tensor(class_weights, dtype=torch.float)

    except Exception as e:
        print(f"\nData loading failed: {str(e)}")
        raise

def construct_graph(features, labels, raw_data):
    try:
        print("\n=== Constructing Graph ===")
        G = nx.Graph()
        node_mapping = {}
        
        numerical_features = features.select_dtypes(include=['number'])
        print(f"Using {len(numerical_features.columns)} numerical features")

        valid_nodes = skipped_nodes = nan_nodes = 0
        
        for idx, row in raw_data.reset_index(drop=True).iterrows():
            try:
                if idx >= len(numerical_features):
                    skipped_nodes += 1
                    continue
                    
                node_id = f"{row['proto']}_{row['service']}_{idx}"
                node_features = numerical_features.iloc[idx].values.astype('float32')
                
                if np.isnan(node_features).any():
                    node_features = np.nan_to_num(node_features)
                    nan_nodes += 1
                    
                if node_id not in node_mapping:
                    node_mapping[node_id] = len(node_mapping)
                    G.add_node(
                        node_mapping[node_id],
                        features=node_features,
                        label=labels[idx]
                    )
                    valid_nodes += 1
                
                target_id = f"dst_{row['dsport']}_{idx}"
                if target_id not in node_mapping:
                    node_mapping[target_id] = len(node_mapping)
                    G.add_node(
                        node_mapping[target_id],
                        features=np.zeros(numerical_features.shape[1], dtype='float32'),
                        label=0
                    )
                
                src, dst = node_mapping[node_id], node_mapping[target_id]
                if G.has_edge(src, dst):
                    G[src][dst]['weight'] += 1
                else:
                    G.add_edge(src, dst, weight=1)
                    
            except Exception:
                skipped_nodes += 1
                continue

        print(f"\n=== Graph Construction Report ===")
        print(f"Valid nodes: {valid_nodes}")
        print(f"Nodes with NaN fixes: {nan_nodes}")
        print(f"Skipped rows: {skipped_nodes}")
        print(f"Total nodes: {len(G.nodes)}")
        print(f"Total edges: {len(G.edges)}")
        
        return G, node_mapping

    except Exception as e:
        print(f"\nGraph construction failed: {str(e)}")
        raise

def convert_to_pyg_format(graph, device='cpu'):
    try:
        print("\n=== Converting to PyG Format ===")
        features = np.array([graph.nodes[n]['features'] for n in graph.nodes], dtype='float32')
        y = torch.tensor([graph.nodes[n]['label'] for n in graph.nodes], dtype=torch.long)
        
        edges = list(graph.edges(data='weight', default=1.0))
        edge_index = torch.tensor([(u, v) for u, v, _ in edges], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([w for _, _, w in edges], dtype=torch.float).view(-1, 1)
        
        print(f"Node features shape: {features.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        
        return Data(
            x=torch.from_numpy(features).to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            y=y.to(device),
            num_nodes=len(graph.nodes)
        )
    except Exception as e:
        print(f"\nPyG conversion failed: {str(e)}")
        raise
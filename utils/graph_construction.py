import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from config import Config
import networkx as nx
import numpy as np
from tqdm import tqdm

def build_graph_from_partition(partition_file):
    try:
        print("Loading data with rigorous index control...")
        data = np.load(partition_file, allow_pickle=True)
        
        src_ips = data['src_ips'].astype('U15')
        dst_ips = data['dst_ips'].astype('U15')
        features = data['features'].astype(np.float32)
        labels = data['labels'].astype(np.int8)

        print("Creating deterministic IP mapping...")
        unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        ip_to_idx = {ip: idx for idx, ip in enumerate(sorted(unique_ips))}
        
        print(f"Total unique IPs: {len(ip_to_idx)}")
        
        # Initialize graph with explicit node creation and feature initialization
        G = nx.Graph()
        feature_shape = features.shape[1]
        
        # Initialize ALL nodes with zero features first
        for ip, idx in ip_to_idx.items():
            G.add_node(idx, 
                      features=np.zeros(feature_shape, dtype=np.float32),
                      label=0)
        
        print("Processing edges with index validation...")
        # Track which nodes actually have features
        nodes_with_features = set()
        
        for i in tqdm(range(len(src_ips)), desc="Edge processing"):
            src_idx = ip_to_idx[src_ips[i]]
            dst_idx = ip_to_idx[dst_ips[i]]
            
            # Validate indices exist in graph
            if not G.has_node(src_idx) or not G.has_node(dst_idx):
                raise ValueError(f"Missing node(s) for edge {src_idx}-{dst_idx}")
            
            # Update node features (average with existing)
            G.nodes[src_idx]['features'] = (G.nodes[src_idx]['features'] * len(nodes_with_features) + features[i]) / (len(nodes_with_features) + 1)
            nodes_with_features.add(src_idx)
            
            G.nodes[src_idx]['label'] = max(G.nodes[src_idx]['label'], labels[i])
            
            # Add edge
            if G.has_edge(src_idx, dst_idx):
                G.edges[src_idx, dst_idx]['weight'] += 1
            else:
                G.add_edge(src_idx, dst_idx, weight=1)

        # Final sanity checks
        print("\nGraph validation report:")
        print(f"Total nodes: {len(G.nodes)}")
        print(f"Node index range: {min(G.nodes)} to {max(G.nodes)}")
        print(f"Total edges: {len(G.edges)}")
        print(f"Nodes with features: {len(nodes_with_features)}")
        
        return G, ip_to_idx

    except Exception as e:
        print(f"Graph construction failed: {str(e)}")
        raise
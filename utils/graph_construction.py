import networkx as nx
import numpy as np
from collections import defaultdict
from config import Config
import gc
from tqdm import tqdm
import sys

def build_graph_from_partition(partition_file):
    """Optimized graph construction with proper memory tracking"""
    try:
        print("Loading partition data...")
        data = np.load(partition_file, allow_pickle=True)
        
        # Validate critical data
        required_arrays = ['features', 'labels', 'src_ips', 'dst_ips']
        for arr in required_arrays:
            if arr not in data:
                raise ValueError(f"Missing required array: {arr}")

        # Convert to memory-efficient formats
        src_ips = data['src_ips'].astype('U15')  # Fixed-width strings
        dst_ips = data['dst_ips'].astype('U15')
        features = data['features'].astype(np.float32)
        labels = data['labels'].astype(np.int8)

        print("Constructing graph (optimized version)...")
        G = nx.Graph()
        ip_mapping = {}
        edge_counts = defaultdict(int)
        
        # Track feature memory usage
        feature_mem = 0
        
        # Process unique IPs
        print("Identifying unique IPs...")
        all_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        
        # Create nodes with initial features
        print("Creating nodes...")
        for ip in tqdm(all_ips, desc="Nodes"):
            ip_mapping[ip] = len(ip_mapping)
            G.add_node(ip_mapping[ip], features=None, label=0)
        
        # Process connections in chunks
        chunk_size = 50000
        num_chunks = int(np.ceil(len(src_ips) / chunk_size))
        
        print("Processing connections...")
        for chunk_idx in tqdm(range(num_chunks), desc="Connections"):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(src_ips))
            
            chunk_src = src_ips[start:end]
            chunk_dst = dst_ips[start:end]
            chunk_feat = features[start:end]
            chunk_labels = labels[start:end]
            
            for i in range(len(chunk_src)):
                src_idx = ip_mapping[chunk_src[i]]
                dst_idx = ip_mapping[chunk_dst[i]]
                
                # Update node features
                if G.nodes[src_idx]['features'] is None:
                    G.nodes[src_idx]['features'] = chunk_feat[i]
                    feature_mem += chunk_feat[i].nbytes
                else:
                    G.nodes[src_idx]['features'] = (
                        G.nodes[src_idx]['features'] + chunk_feat[i]
                    ) / 2
                
                # Update label
                G.nodes[src_idx]['label'] = max(
                    G.nodes[src_idx]['label'], 
                    chunk_labels[i]
                )
                
                # Track edges
                edge_key = tuple(sorted((src_idx, dst_idx)))
                edge_counts[edge_key] += 1
            
            # Clear memory every 10 chunks
            if chunk_idx % 10 == 0:
                gc.collect()
        
        # Add edges with weights
        print("Creating edges...")
        for (u, v), weight in tqdm(edge_counts.items(), desc="Edges"):
            G.add_edge(u, v, weight=weight)
        
        # Final validation
        print("Finalizing graph...")
        for n in G.nodes():
            if G.nodes[n]['features'] is None:
                G.nodes[n]['features'] = np.zeros(features.shape[1], dtype=np.float32)
                feature_mem += G.nodes[n]['features'].nbytes
        
        print("\n=== Optimized Graph Stats ===")
        print(f"Nodes: {len(G.nodes):,}")
        print(f"Edges: {len(G.edges):,}")
        print(f"Feature memory: {feature_mem/(1024**2):.2f} MB")
        print(f"Graph memory: {sys.getsizeof(G)/(1024**2):.2f} MB (estimated)")
        
        return G, ip_mapping
        
    except Exception as e:
        print(f"\nGraph construction failed: {str(e)}")
        raise
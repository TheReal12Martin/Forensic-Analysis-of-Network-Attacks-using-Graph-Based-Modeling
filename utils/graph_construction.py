import networkx as nx
import numpy as np
from tqdm import tqdm
from config import Config
import gc

def build_graph_from_partition(partition_file):
    """Build graph with memory limits using Config"""
    try:
        # Load with validation
        data = np.load(partition_file)
        required = ['features', 'labels', 'src_ips', 'dst_ips']
        if not all(k in data for k in required):
            raise KeyError(f"Partition missing required keys: {required}")

        # Apply sampling if needed
        n_samples = len(data['features'])
        sample_size = min(n_samples, Config.MAX_GRAPH_NODES * 2)
        idx = np.random.choice(n_samples, sample_size, replace=False)
        
        features = data['features'][idx]
        src_ips = data['src_ips'][idx]
        dst_ips = data['dst_ips'][idx]

        # Build graph with progress tracking
        G = nx.Graph()
        ip_to_idx = {}
        edge_count = 0
        
        print(f"Building graph (max {Config.MAX_GRAPH_NODES} nodes)...")
        for i, (src, dst, feat) in tqdm(enumerate(zip(src_ips, dst_ips, features)), 
                                       total=len(src_ips)):
            if len(ip_to_idx) >= Config.MAX_GRAPH_NODES:
                break
            if edge_count >= Config.MAX_GRAPH_EDGES:
                break

            # Node management
            for ip in [src, dst]:
                if ip not in ip_to_idx:
                    ip_to_idx[ip] = len(ip_to_idx)
                    G.add_node(ip_to_idx[ip], features=feat)

            # Edge management
            src_idx, dst_idx = ip_to_idx[src], ip_to_idx[dst]
            if G.has_edge(src_idx, dst_idx):
                G[src_idx][dst_idx]['weight'] += 1
            else:
                G.add_edge(src_idx, dst_idx, weight=1)
                edge_count += 1

        print(f"Built graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G, ip_to_idx

    except Exception as e:
        print(f"Graph construction failed: {str(e)}")
        raise
    finally:
        gc.collect()
import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from config import Config
import gc

def build_graph_from_partition(partition_file):
    """Build graph only from valid partitions"""
    try:
        data = np.load(partition_file)
        
        # Verify both classes exist
        unique_labels = np.unique(data['labels'])
        if len(unique_labels) < 2:
            print(f"⚠️ Skipping {os.path.basename(partition_file)} - only class {unique_labels[0]} present")
            return None, None

        # Verify minimum samples
        if len(data['labels']) < 1000:
            print(f"⚠️ Skipping {os.path.basename(partition_file)} - only {len(data['labels'])} samples")
            return None, None

        # Sampling with class balance
        benign_idx = np.where(data['labels'] == 0)[0]
        malicious_idx = np.where(data['labels'] == 1)[0]
        
        sample_size = min(
            Config.MAX_GRAPH_NODES // 2,
            len(benign_idx),
            len(malicious_idx)
        )
        
        idx = np.concatenate([
            np.random.choice(benign_idx, sample_size, replace=False),
            np.random.choice(malicious_idx, sample_size, replace=False)
        ])

        # Graph construction
        G = nx.Graph()
        ip_to_idx = {}
        
        for i in tqdm(idx, desc="Building graph"):
            src = data['src_ips'][i]
            dst = data['dst_ips'][i]
            feat = data['features'][i]
            label = data['labels'][i]

            # Add nodes
            for ip in [src, dst]:
                if ip not in ip_to_idx:
                    ip_to_idx[ip] = len(ip_to_idx)
                    G.add_node(ip_to_idx[ip], features=feat, label=label)

            # Add edge
            if G.has_edge(ip_to_idx[src], ip_to_idx[dst]):
                G[ip_to_idx[src]][ip_to_idx[dst]]['weight'] += 1
            else:
                G.add_edge(ip_to_idx[src], ip_to_idx[dst], weight=1)

        print(f"✅ Built graph from {os.path.basename(partition_file)} "
              f"({len(G.nodes)} nodes, {len(G.edges)} edges) "
              f"Class balance: {dict(zip(*np.unique(data['labels'][idx], return_counts=True)))}")
        return G, ip_to_idx

    except Exception as e:
        print(f"❌ Graph construction failed: {str(e)}")
        return None, None
    finally:
        gc.collect()
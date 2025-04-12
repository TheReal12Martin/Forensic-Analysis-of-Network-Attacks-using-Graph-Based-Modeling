import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from config import Config
import gc

def build_graph_from_partition(partition_file):
    try:
        # Memory-mapped loading with manual chunking
        data = np.load(partition_file, mmap_mode='r')
        
        # Get minimum samples (reduced further)
        min_samples = min(
            len(np.where(data['labels'] == 0)[0]),
            len(np.where(data['labels'] == 1)[0]),
            Config.MAX_GRAPH_NODES // 4  # Only use 25% of max nodes
        )
        
        # Select indices
        idx0 = np.random.choice(
            np.where(data['labels'] == 0)[0], 
            min_samples, 
            replace=False
        )
        idx1 = np.random.choice(
            np.where(data['labels'] == 1)[0], 
            min_samples, 
            replace=False
        )
        selected_idx = np.concatenate([idx0, idx1])

        # Build tiny graph
        G = nx.Graph()
        features = []
        
        # Process in nano-batches
        for i in range(0, len(selected_idx), 100):  # 100-sample chunks
            chunk = selected_idx[i:i+100]
            chunk_data = data['features'][chunk]
            for j, idx in enumerate(chunk):
                node_id = i + j  # Simple 0-based indexing
                G.add_node(node_id, features=chunk_data[j], label=data['labels'][idx])
                features.append(chunk_data[j])
                
                if G.number_of_nodes() >= Config.MAX_GRAPH_NODES:
                    break

        # Tiny edge construction (very sparse)
        if len(features) > 1:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(features)
            distances, indices = nbrs.kneighbors(features)
            
            for i, (dists, neighbors) in enumerate(zip(distances, indices)):
                for j, d in zip(neighbors, dists):
                    if i != j and d < 0.9:  # Strict similarity threshold
                        G.add_edge(i, j, weight=1-d)
                        if G.number_of_edges() >= Config.MAX_GRAPH_EDGES:
                            break

        print(f"Built micro-graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, None

    except Exception as e:
        print(f"Graph build failed: {str(e)}")
        return None, None
    finally:
        if 'data' in locals():
            del data
        gc.collect()
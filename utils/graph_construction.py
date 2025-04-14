import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from config import Config
import gc

def build_graph_from_partition(partition_file):
    try:
        data = np.load(partition_file, mmap_mode='r')
        
        # Dynamic sampling based on class balance
        class_0_idx = np.where(data['labels'] == 0)[0]
        class_1_idx = np.where(data['labels'] == 1)[0]
        min_samples = min(len(class_0_idx), len(class_1_idx), Config.MAX_GRAPH_NODES//2)
        
        # Stratified sampling with feature diversity
        def get_diverse_samples(indices, k):
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE)
            kmeans.fit(data['features'][indices])
            _, sample_indices = np.unique(kmeans.labels_, return_index=True)
            return indices[sample_indices]
        
        selected_idx = np.concatenate([
            get_diverse_samples(class_0_idx, min_samples//2),
            get_diverse_samples(class_1_idx, min_samples//2)
        ])

        # Enhanced edge construction with feature similarity
        G = nx.Graph()
        features = data['features'][selected_idx]
        
        for i, idx in enumerate(selected_idx):
            G.add_node(i, features=features[i], label=data['labels'][idx])
        
        # Improved similarity edges
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        for i, (dists, neighbors) in enumerate(zip(distances, indices)):
            for j, d in zip(neighbors, dists):
                if i != j and d < 0.7:  # Stricter similarity threshold
                    weight = 1 - (d * (1 + G.degree(i)/10))  # Degree-adjusted weight
                    G.add_edge(i, j, weight=weight)
                    
        print(f"Built enhanced graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, None
        
    except Exception as e:
        print(f"Graph build failed: {str(e)}")
        return None, None
    finally:
        if 'data' in locals():
            del data
        gc.collect()
import os
import networkx as nx
import numpy as np
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from config import Config
import gc

def build_graph_from_partition(partition_file):
    try:
        data = np.load(partition_file, mmap_mode='r')
        
        # Enforce minimum samples per class
        class_0_idx = np.where(data['labels'] == 0)[0]
        class_1_idx = np.where(data['labels'] == 1)[0]
        
        min_class_samples = max(Config.MIN_SAMPLES_PER_CLASS, 
                               Config.MIN_NEIGHBORS + 1)  # +1 for self
        min_samples = min(len(class_0_idx), len(class_1_idx))
        
        if min_samples < min_class_samples:
            print(f"Insufficient samples (class 0: {len(class_0_idx)}, class 1: {len(class_1_idx)})")
            return None, None

        # Stratified sampling with feature diversity
        def get_diverse_samples(indices, k):
            from sklearn.cluster import KMeans
            actual_k = min(k, len(indices))
            if actual_k < 2:
                return indices[:actual_k]
                
            kmeans = KMeans(n_clusters=actual_k, random_state=Config.RANDOM_STATE)
            try:
                kmeans.fit(data['features'][indices])
                _, sample_indices = np.unique(kmeans.labels_, return_index=True)
                return indices[sample_indices]
            except NotFittedError:
                return indices[:actual_k]

        selected_idx = np.concatenate([
            get_diverse_samples(class_0_idx, min(Config.MAX_GRAPH_NODES//2, len(class_0_idx))),
            get_diverse_samples(class_1_idx, min(Config.MAX_GRAPH_NODES//2, len(class_1_idx)))
        ])

        # Build graph with dynamic neighbor selection
        G = nx.Graph()
        features = data['features'][selected_idx]
        
        for i, idx in enumerate(selected_idx):
            G.add_node(i, features=features[i], label=data['labels'][idx])
        
        # Dynamic edge construction
        n_samples = len(selected_idx)
        n_neighbors = min(Config.MAX_NEIGHBORS, n_samples - 1)
        n_neighbors = max(n_neighbors, Config.MIN_NEIGHBORS)
        
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(features)
            distances, indices = nbrs.kneighbors(features)
            
            for i, (dists, neighbors) in enumerate(zip(distances, indices)):
                for j, d in zip(neighbors, dists):
                    if i != j and d < Config.MIN_SIMILARITY:
                        weight = 1 - (d * (1 + G.degree(i)/10)) 
                        G.add_edge(i, j, weight=weight)
        except ValueError as e:
            print(f"Edge construction failed: {str(e)}")
            return None, None
            
        print(f"Built enhanced graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, None
        
    except Exception as e:
        print(f"Graph build failed: {str(e)}")
        return None, None
    finally:
        if 'data' in locals():
            del data
        gc.collect()
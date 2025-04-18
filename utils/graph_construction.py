import os
import networkx as nx
import numpy as np
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from config import Config
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

def _safe_load_data(partition_file):
    """Load data safely into memory for modification"""
    with np.load(partition_file) as data:
        return {
            'features': data['features'].copy(),
            'labels': data['labels'].copy(),
            'src_ips': data['src_ips'].copy() if 'src_ips' in data else None,
            'dst_ips': data['dst_ips'].copy() if 'dst_ips' in data else None
        }

def _handle_imbalance(features, labels):
    """Apply imbalance correction with safety checks"""
    # Calculate current class distribution
    class_counts = np.bincount(labels)
    minority_class = 1 if class_counts[1] < class_counts[0] else 0
    minority_count = class_counts[minority_class]
    
    if Config.SMOTE_ENABLED and minority_count >= 2:
        try:
            smote = SMOTE(
                sampling_strategy=min(0.5, Config.IMBALANCE_THRESHOLD/10),
                random_state=Config.RANDOM_STATE,
                k_neighbors=min(Config.SMOTE_K_NEIGHBORS, minority_count-1)
            )
            features_res, labels_res = smote.fit_resample(features, labels)
            print(f"Generated {sum(labels_res == 1)} malicious samples via SMOTE")
            return features_res, labels_res
        except Exception as e:
            print(f"SMOTE failed: {str(e)} - using random oversampling")
    
    # Random oversampling fallback
    minority_idx = np.where(labels == minority_class)[0]
    n_samples = min(
        int(max(class_counts)/Config.IMBALANCE_THRESHOLD),
        Config.MAX_GRAPH_NODES//2
    )
    oversampled_idx = resample(
        minority_idx,
        replace=True,
        n_samples=n_samples,
        random_state=Config.RANDOM_STATE
    )
    return (
        np.vstack([features, features[oversampled_idx]]),
        np.concatenate([labels, labels[oversampled_idx]])
    )

def build_graph_from_partition(partition_file):
    """Build graph from partition data with proper imbalance handling"""
    try:
        # 1. Load data safely
        data = _safe_load_data(partition_file)
        features = data['features']
        labels = data['labels']
        
        # 2. Calculate and handle imbalance
        class_counts = np.bincount(labels)
        imbalance_ratio = max(class_counts[0]/max(1,class_counts[1]), 
                          class_counts[1]/max(1,class_counts[0]))
        
        if imbalance_ratio > Config.IMBALANCE_THRESHOLD:
            print(f"⚠️ Extreme imbalance ({imbalance_ratio:.1f}:1) - applying correction")
            features, labels = _handle_imbalance(features, labels)
            class_counts = np.bincount(labels)
            imbalance_ratio = max(class_counts[0]/max(1,class_counts[1]), 
                              class_counts[1]/max(1,class_counts[0]))
            print(f"→ New imbalance ratio: {imbalance_ratio:.1f}:1")

        # 3. Validate minimum samples
        if any(c < Config.MIN_SAMPLES_PER_CLASS for c in class_counts):
            print(f"❌ Insufficient samples (class 0: {class_counts[0]}, class 1: {class_counts[1]})")
            return None, None

        # 4. Select diverse samples
        def _select_diverse_samples(features, labels, max_samples):
            selected_idx = []
            for class_label in [0, 1]:
                class_idx = np.where(labels == class_label)[0]
                if len(class_idx) == 0:
                    continue
                    
                n_samples = min(max_samples, len(class_idx))
                if n_samples <= Config.MIN_NEIGHBORS + 1:
                    selected = class_idx[:n_samples]
                else:
                    kmeans = KMeans(
                        n_clusters=min(n_samples, 50),
                        random_state=Config.RANDOM_STATE
                    ).fit(features[class_idx])
                    _, cluster_rep_idx = np.unique(kmeans.labels_, return_index=True)
                    selected = class_idx[cluster_rep_idx]
                    
                    if len(selected) < n_samples:
                        remaining = np.setdiff1d(class_idx, selected)
                        selected = np.concatenate([
                            selected,
                            np.random.choice(remaining, n_samples-len(selected), replace=False)
                        ])
                selected_idx.append(selected)
            return np.concatenate(selected_idx)

        selected_idx = _select_diverse_samples(
            features, 
            labels,
            Config.MAX_GRAPH_NODES//2
        )
        selected_features = features[selected_idx]
        selected_labels = labels[selected_idx]

        # 5. Build graph
        G = nx.Graph()
        for i, (feat, label) in enumerate(zip(selected_features, selected_labels)):
            G.add_node(i, features=feat, label=label)

        # 6. Create edges
        if len(selected_features) > 1:
            n_neighbors = min(
                Config.MAX_NEIGHBORS,
                len(selected_features) - 1
            )
            n_neighbors = max(n_neighbors, Config.MIN_NEIGHBORS)
            
            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine'
            ).fit(selected_features)
            
            distances, indices = nbrs.kneighbors(selected_features)
            for i, (dists, neighbors) in enumerate(zip(distances, indices)):
                for j, d in zip(neighbors, dists):
                    if i != j and d < Config.MIN_SIMILARITY:
                        weight = 1 - (d * (1 + G.degree(i)/10)) 
                        G.add_edge(i, j, weight=weight)

        print(f"✅ Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, None

    except Exception as e:
        print(f"❌ Graph build failed: {str(e)}")
        return None, None
    finally:
        gc.collect()
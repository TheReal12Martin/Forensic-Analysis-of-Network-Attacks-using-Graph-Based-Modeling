import networkx as nx
import numpy as np
from config import Config
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from imblearn.over_sampling import ADASYN, SMOTE

def _safe_load_data(partition_file):
    """Load data safely into memory for modification"""
    with np.load(partition_file) as data:
        return {
            'features': data['features'].copy(),
            'labels': data['labels'].copy(),
            'src_ips': data['src_ips'].copy() if 'src_ips' in data else None,
            'dst_ips': data['dst_ips'].copy() if 'dst_ips' in data else None
        }

def _apply_smote(features, labels, minority_count):
    smote = SMOTE(
        sampling_strategy=min(0.5, Config.IMBALANCE_THRESHOLD/10),
        k_neighbors=min(Config.SMOTE_NEIGHBORS, minority_count-1),
        random_state=Config.RANDOM_STATE
    )
    return smote.fit_resample(features, labels)

def _apply_adasyn(features, labels, minority_count):
    ada = ADASYN(
        sampling_strategy=min(0.5, Config.IMBALANCE_THRESHOLD/5),
        n_neighbors=min(Config.ADASYN_NEIGHBORS, minority_count-1),
        random_state=Config.RANDOM_STATE
    )
    return ada.fit_resample(features, labels)

def _handle_imbalance(features, labels):
    """Hybrid imbalance correction with auto-selection"""
    class_counts = np.bincount(labels)
    minority_class = 1 if class_counts[1] < class_counts[0] else 0
    minority_count = class_counts[minority_class]
    imbalance_ratio = max(class_counts)/max(1, minority_count)
    
    try:
        if Config.IMBALANCE_METHOD == 'auto':
            if imbalance_ratio > Config.AUTO_SMOTE_THRESHOLD:
                print("⚡ Using SMOTE for extreme imbalance")
                return _apply_smote(features, labels, minority_count)
            else:
                print("⚡ Using ADASYN for moderate imbalance")
                return _apply_adasyn(features, labels, minority_count)
        elif Config.IMBALANCE_METHOD == 'smote':
            return _apply_smote(features, labels, minority_count)
        elif Config.IMBALANCE_METHOD == 'adasyn':
            return _apply_adasyn(features, labels, minority_count)
    except Exception as e:
        print(f"⚠️ {Config.IMBALANCE_METHOD} failed: {str(e)} - using random oversampling")
    
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
        features, labels = data['features'], data['labels']
        
        # Calculate initial imbalance
        class_counts = np.bincount(labels)
        imbalance_ratio = max(class_counts)/max(1, min(class_counts))
        
        if imbalance_ratio > Config.IMBALANCE_THRESHOLD:
            print(f"⚠️ Imbalance {imbalance_ratio:.1f}:1 - applying correction")
            features, labels = _handle_imbalance(features, labels)
            new_counts = np.bincount(labels)
            print(f"→ New distribution: Class 0={new_counts[0]}, Class 1={new_counts[1]}")

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
from collections import defaultdict
import networkx as nx
import community as community_louvain
from sklearn.cluster import SpectralClustering
from typing import Dict, Tuple

class GraphAnalyzer:
    def __init__(self):
        self.community_algorithms = {
            'louvain': self.detect_communities_louvain,
            'spectral': self.detect_communities_spectral,
            'girvan_newman': self.detect_communities_girvan_newman
        }
    
    def detect_communities(self, graph_data, algorithm: str = 'louvain', min_community_size=3, **kwargs):
        """Improved unified community detection"""
        G = self._create_networkx_graph(graph_data)
        
        print(f"\nDetecting communities using {algorithm} on graph with:")
        print(f"- Nodes: {len(G.nodes())}")
        print(f"- Edges: {len(G.edges())}")
        print(f"- Density: {nx.density(G):.4f}")
        
        if algorithm not in self.community_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Run the selected algorithm
        result = self.community_algorithms[algorithm](G, **kwargs)
        
        # Post-process all results to handle small communities
        return self._postprocess_communities(G, result, min_community_size)

    def _postprocess_communities(self, G, result, min_community_size):
        """Common post-processing for all algorithms"""
        partition = result['partition']
        
        # Analyze initial communities
        comm_sizes = {}
        for node, comm in partition.items():
            comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
        
        print(f"Initial communities: {len(comm_sizes)}")
        print("Top communities:", sorted(comm_sizes.items(), key=lambda x: -x[1])[:5])
        
        # Recalculate metrics
        result['partition'] = partition
        result['num_communities'] = len(set(partition.values()))
        if 'modularity' in result:
            result['modularity'] = community_louvain.modularity(partition, G)
        
        return result

    def detect_communities_louvain(self, G, resolution=1.0):
        """Improved Louvain with resolution control"""
        partition = community_louvain.best_partition(G, resolution=resolution)
        return {
            'algorithm': 'louvain',
            'partition': partition,
            'modularity': community_louvain.modularity(partition, G)
        }

    def detect_communities_spectral(self, G, n_clusters=None):
        """Spectral clustering with automatic cluster count"""
        if n_clusters is None:
            n_clusters = max(2, min(10, int(len(G.nodes()) ** 0.3)))
        
        adj_matrix = nx.to_numpy_array(G)
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='discretize'
        ).fit(adj_matrix)
        
        partition = {node: int(label) for node, label in zip(G.nodes(), clustering.labels_)}
        return {
            'algorithm': 'spectral',
            'partition': partition,
            'num_communities': n_clusters
        }

    def detect_communities_girvan_newman(self, G):
        """Girvan-Newman with early stopping"""
        communities_generator = nx.algorithms.community.girvan_newman(G)
        
        # Get first level (2 communities)
        first_level = next(communities_generator)
        
        # Get second level if graph is large enough
        if len(G.nodes()) > 50:
            next_level = next(communities_generator)
            communities = sorted(map(sorted, next_level))
        else:
            communities = sorted(map(sorted, first_level))
        
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        return {
            'algorithm': 'girvan_newman',
            'partition': partition,
            'num_communities': len(communities)
        }
    
    def _create_networkx_graph(self, graph_data: Dict) -> nx.Graph:
        """More robust graph creation"""
        G = nx.Graph()
        
        # Validate input
        if not isinstance(graph_data, dict):
            raise ValueError("graph_data must be a dictionary")
        
        # Add nodes
        if 'nodes' not in graph_data:
            raise ValueError("graph_data must contain 'nodes' key")
        
        if isinstance(graph_data['nodes'], (list, tuple)):
            G.add_nodes_from(graph_data['nodes'])
        elif isinstance(graph_data['nodes'], dict):
            G.add_nodes_from(graph_data['nodes'].items())
        else:
            raise ValueError("nodes must be list or dict")
        
        # Add edges
        if 'edges' in graph_data:
            if isinstance(graph_data['edges'], (list, tuple)) and len(graph_data['edges']) == 2:
                # Assume edges are given as [sources], [targets]
                for src, tgt in zip(graph_data['edges'][0], graph_data['edges'][1]):
                    if src < len(graph_data['nodes']) and tgt < len(graph_data['nodes']):
                        G.add_edge(graph_data['nodes'][src], graph_data['nodes'][tgt])
            elif isinstance(graph_data['edges'], list):
                # Assume edges are given as list of pairs
                for edge in graph_data['edges']:
                    if len(edge) == 2:
                        G.add_edge(edge[0], edge[1])
        
        print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    def get_community_metrics(self, G, partition: Dict) -> Dict:
        """Calculate metrics for the communities"""
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Convert lists to tuples for dictionary keys
        inter_edges = self._calculate_inter_community_edges(G, partition)
        inter_edges_fixed = {str(k): v for k, v in inter_edges.items()}  # Convert tuple keys to strings
        
        metrics = {
            'community_sizes': {str(k): len(v) for k, v in communities.items()},  # Ensure keys are strings
            'inter_community_edges': inter_edges_fixed,
            'intra_community_edges': {str(k): v for k, v in self._calculate_intra_community_edges(G, partition).items()},
            'community_density': {str(k): v for k, v in self._calculate_community_density(G, communities).items()}
        }
        
        return metrics

    def _calculate_inter_community_edges(self, G, partition) -> Dict[Tuple, int]:
        """Count edges between different communities"""
        inter_edges = {}
        for u, v in G.edges():
            if partition[u] != partition[v]:
                key = tuple(sorted((partition[u], partition[v])))
                inter_edges[key] = inter_edges.get(key, 0) + 1
        return inter_edges
    
    def _calculate_intra_community_edges(self, G, partition) -> Dict[int, int]:
        """Count edges within the same community"""
        intra_edges = {}
        for u, v in G.edges():
            if partition[u] == partition[v]:
                comm = partition[u]
                intra_edges[comm] = intra_edges.get(comm, 0) + 1
        return intra_edges
    
    def _calculate_community_density(self, G, communities) -> Dict[int, float]:
        """Calculate density of each community"""
        densities = {}
        for comm_id, nodes in communities.items():
            subgraph = G.subgraph(nodes)
            densities[comm_id] = nx.density(subgraph)
        return densities
    
    def detect_attack_campaigns(self, G, partition, predictions):
        """Identify communities with high attack node concentration"""
        attack_communities = {}
        for node, comm in partition.items():
            if predictions.get(node, 0) == 1:  
                attack_communities[comm] = attack_communities.get(comm, 0) + 1
        
        # Normalize by community size
        community_sizes = {comm: sum(1 for n in partition.values() if n == comm) 
                        for comm in set(partition.values())}
        
        return {
            comm: (count, count/community_sizes[comm])
            for comm, count in attack_communities.items()
            if count/community_sizes[comm] > 0.7  # Threshold for "high concentration"
        }

    def detect_lateral_movement(self, G, partition):
        """Find nodes connecting different communities"""
        bridge_nodes = []
        # Compute betweenness once
        betweenness = nx.betweenness_centrality(G)

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                continue
            
            neighbor_comms = {partition[n] for n in neighbors if n in partition}
            if len(neighbor_comms) > 1:
                bridge_nodes.append({
                    'node': node,
                    'communities_connected': len(neighbor_comms),
                    'betweenness': betweenness.get(node, 0)
                })
        
        return sorted(bridge_nodes, key=lambda x: -x['betweenness'])
    

    def detect_command_control(self, G, partition):
        """Identify star-shaped communities with central nodes"""
        suspicious = []
        # Pre-group nodes by community
        comm_nodes = defaultdict(list)
        for node, comm in partition.items():
            comm_nodes[comm].append(node)

        for comm, nodes in comm_nodes.items():
            if len(nodes) < 3:
                continue

            subgraph = G.subgraph(nodes)
            degrees = dict(subgraph.degree())
            max_degree = max(degrees.values())
            
            if max_degree / len(nodes) > 0.8:
                center = max(degrees.items(), key=lambda x: x[1])[0]
                suspicious.append({
                    'community': comm,
                    'center_node': center,
                    'degree': max_degree,
                    'size': len(nodes)
                })
        
        return suspicious
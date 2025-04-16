import tempfile
import time
import pyshark
import numpy as np
from collections import defaultdict
import torch
import warnings
from typing import Dict, Optional, List

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PCAPProcessor:
    def __init__(self):
        self.flows = defaultdict(list)
        self.node_features = {}
        self.edge_connections = set()
        self.packet_count = 0
        self.last_report = time.time()
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp()  # Store large data on disk

    def process_pcap(self, pcap_path: str, max_packets: Optional[int] = None) -> Optional[Dict]:
        """Process PCAP file into graph data with 13 features per node"""
        print(f"\nStarting PCAP processing: {pcap_path}")
        print("=" * 60)
        
        try:
            cap = pyshark.FileCapture(
                pcap_path,
                display_filter='ip and (tcp or udp or icmp)',
                keep_packets=False,
                use_json=True,
                include_raw=False
            )
            
            # Process packets in chunks
            chunk_size = 100000
            processed_in_chunk = 0
            
            for pkt in cap:
                if max_packets and self.packet_count >= max_packets:
                    print(f"Reached maximum packet limit ({max_packets})")
                    break
                    
                self._process_packet(pkt)
                self.packet_count += 1
                processed_in_chunk += 1
                
                if self.packet_count % 50000 == 0:
                    self._report_progress()
                
                if processed_in_chunk >= chunk_size:
                    processed_in_chunk = 0
                    self._cleanup_temp_resources()
            
            self._calculate_flow_stats()
            graph_data = self._build_graph_data()
            
            # Print summary
            self._print_summary()
            
            return graph_data
            
        except Exception as e:
            print(f"\nPCAP processing failed: {str(e)}")
            return None
        finally:
            if 'cap' in locals():
                cap.close()
            self._cleanup_temp_resources()

    def _report_progress(self):
        """Report processing progress"""
        current_time = time.time()
        if current_time - self.last_report > 5:
            elapsed = current_time - self.start_time
            rate = self.packet_count / elapsed if elapsed > 0 else 0
            print(f"Processed {self.packet_count} packets "
                  f"({rate:.2f} packets/sec, {elapsed:.2f} sec elapsed)")
            self.last_report = current_time

    def _cleanup_temp_resources(self):
        """Clean up temporary resources"""
        try:
            if hasattr(pyshark.capture.capture, '_current_tshark'):
                del pyshark.capture.capture._current_tshark
            import gc
            gc.collect()
        except:
            pass

    def _process_packet(self, pkt):
        """Process individual network packet"""
        try:
            if not hasattr(pkt, 'ip') or not hasattr(pkt.ip, 'src') or not hasattr(pkt.ip, 'dst'):
                return
                
            src = str(pkt.ip.src)
            dst = str(pkt.ip.dst)
            
            proto = None
            if hasattr(pkt, 'tcp'):
                proto = 'tcp'
            elif hasattr(pkt, 'udp'):
                proto = 'udp'
            elif hasattr(pkt, 'icmp'):
                proto = 'icmp'
            
            if proto:
                flow_key = (src, dst, proto)
                features = {
                    'timestamp': float(pkt.sniff_timestamp) if hasattr(pkt, 'sniff_timestamp') else 0.0,
                    'size': int(pkt.length) if hasattr(pkt, 'length') else 0,
                    'flags': self._get_flags(pkt),
                    'duration': 0
                }
                self.flows[flow_key].append(features)
                self._update_graph_entities(src, dst)
                
        except Exception:
            pass

    def _get_flags(self, pkt) -> int:
        """Extract protocol flags safely"""
        try:
            if hasattr(pkt, 'tcp') and hasattr(pkt.tcp, 'flags'):
                return int(pkt.tcp.flags, 16)
            elif hasattr(pkt, 'icmp') and hasattr(pkt.icmp, 'type'):
                return int(pkt.icmp.type)
            return 0
        except:
            return 0

    def _update_graph_entities(self, src: str, dst: str):
        """Update graph nodes and edges"""
        for node in [src, dst]:
            if node not in self.node_features:
                self.node_features[node] = {
                    'connections': 0,
                    'type': 'host',
                    'first_seen': time.time()
                }
            self.node_features[node]['connections'] += 1
        
        edge_key = tuple(sorted((src, dst)))
        if edge_key not in self.edge_connections:
            self.edge_connections.add(edge_key)

    def _calculate_flow_stats(self):
        """Calculate temporal flow statistics"""
        for flow_key, packets in self.flows.items():
            if len(packets) > 1:
                timestamps = [p['timestamp'] for p in packets if 'timestamp' in p]
                if timestamps:
                    flow_duration = max(timestamps) - min(timestamps)
                    for p in packets:
                        p['duration'] = flow_duration

    def _build_graph_data(self) -> Optional[Dict]:
        """Build graph with 13 features per node"""
        if not self.node_features:
            print("No valid nodes found in PCAP")
            return None
            
        features = []
        node_names = list(self.node_features.keys())
        
        for node in node_names:
            # Basic features
            connections = self.node_features[node]['connections']
            flow_count = sum(1 for flow in self.flows if node in flow[:2])
            is_host = 1 if self.node_features[node]['type'] == 'host' else 0
            time_active = time.time() - self.node_features[node]['first_seen']
            
            # Advanced features
            flow_sizes = [p['size'] for flow in self.flows.values() for p in flow if node in flow[:2]]
            flow_durations = [p['duration'] for flow in self.flows.values() for p in flow if node in flow[:2]]
            flag_counts = [p['flags'] for flow in self.flows.values() for p in flow if node in flow[:2]]
            
            # 13-dimensional feature vector
            features.append([
                connections,                     # Feature 1
                flow_count,                      # Feature 2
                is_host,                         # Feature 3
                time_active,                     # Feature 4
                len(flow_sizes),                 # Feature 5
                np.mean(flow_sizes) if flow_sizes else 0,      # Feature 6
                np.std(flow_sizes) if flow_sizes else 0,       # Feature 7
                np.mean(flow_durations) if flow_durations else 0, # Feature 8
                np.std(flow_durations) if flow_durations else 0,  # Feature 9
                len(flag_counts),                # Feature 10
                np.mean(flag_counts) if flag_counts else 0,    # Feature 11
                np.std(flag_counts) if flag_counts else 0,     # Feature 12
                int(any(f > 0 for f in flag_counts))          # Feature 13
            ])
        
        # Edge construction
        edge_index = []
        edge_attr = []
        
        for src, dst in self.edge_connections:
            src_idx = node_names.index(src)
            dst_idx = node_names.index(dst)
            edge_index.append([src_idx, dst_idx])
            weight = sum(1 for flow in self.flows if (src in flow[:2] and dst in flow[:2]))
            edge_attr.append(weight)
        
        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [0.0]
        
        return {
            'nodes': node_names,
            'x': torch.tensor(features, dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float),
            'y': torch.zeros(len(node_names), dtype=torch.long)
        }

    def _print_summary(self):
        """Print processing summary"""
        processing_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"Processing complete")
        print(f"Total packets processed: {self.packet_count}")
        print(f"Total nodes created: {len(self.node_features)}")
        print(f"Total edges created: {len(self.edge_connections)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Processing rate: {self.packet_count/processing_time:.2f} packets/sec")
        print("=" * 60 + "\n")
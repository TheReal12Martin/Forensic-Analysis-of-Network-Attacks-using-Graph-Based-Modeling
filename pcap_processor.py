import tempfile
import time
import pyshark
import numpy as np
from collections import defaultdict
import torch
import warnings
from typing import Dict, Optional, List
from multiprocessing import Pool, cpu_count
import subprocess
import csv
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

class PCAPProcessor:
    def __init__(self):
        self.flows = defaultdict(list)
        self.node_features = {}
        self.edge_connections = set()
        self.packet_count = 0
        self.last_report = time.time()
        self.start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        print(f"Initialized processor | Device: {self.device} | Temp dir: {self.temp_dir}")

    def process_pcap(self, pcap_path: str, max_packets: Optional[int] = None) -> Optional[Dict]:
        """Main processing with optimized memory handling"""
        print(f"\n📦 Processing {os.path.basename(pcap_path)} (max packets: {max_packets or 'unlimited'})")
        
        try:
            # Try faster tshark CLI first
            try:
                return self._process_via_tshark_cli(pcap_path, max_packets)
            except Exception as e:
                print(f"⚠️ tshark CLI failed, falling back to pyshark: {str(e)}")
                return self._process_via_pyshark(pcap_path, max_packets)
        except Exception as e:
            print(f"❌ PCAP processing failed: {str(e)}")
            return None
        finally:
            self._cleanup_temp_resources()

    def _process_via_pyshark(self, pcap_path: str, max_packets: Optional[int]) -> Optional[Dict]:
        """Fallback to pyshark with memory optimizations"""
        cap = pyshark.FileCapture(
            pcap_path,
            display_filter='ip and (tcp or udp or icmp) and !dns',  # Filter non-essential traffic
            keep_packets=False,  # Critical for memory
            use_json=True,
            include_raw=False
        )
        
        chunk_size = 100000
        processed_in_chunk = 0
        
        for pkt in cap:
            if max_packets and self.packet_count >= max_packets:
                break
                
            self._process_packet(pkt)
            self.packet_count += 1
            processed_in_chunk += 1
            
            if processed_in_chunk >= chunk_size:
                self._cleanup_temp_resources()
                processed_in_chunk = 0
                
        cap.close()
        return self._finalize_graph()

    def _process_via_tshark_cli(self, pcap_path: str, max_packets: Optional[int]) -> Optional[Dict]:
        """Faster processing using tshark CLI"""
        csv_path = os.path.join(self.temp_dir, "packets.csv")
        
        cmd = [
            'tshark', '-r', pcap_path,
            '-T', 'fields',
            '-E', 'header=y',
            '-E', 'separator=,',
            '-e', 'ip.src',
            '-e', 'ip.dst',
            '-e', 'frame.time_epoch',
            '-e', 'frame.len',
            '-e', 'tcp.flags',
            '-e', 'udp.length',
            '-e', 'icmp.type',
            '-Y', 'ip and (tcp or udp or icmp) and !dns'  # Same filter as pyshark
        ]
        
        if max_packets:
            cmd.extend(['-c', str(max_packets)])
        
        with open(csv_path, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        
        # Process CSV in chunks
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._process_csv_row(row)
                self.packet_count += 1
                
        return self._finalize_graph()

    def _process_csv_row(self, row: Dict):
        """Process a single CSV row from tshark output"""
        src = row.get('ip.src', '').strip()
        dst = row.get('ip.dst', '').strip()
        if not src or not dst:
            return
            
        proto = 'tcp' if 'tcp.flags' in row else 'udp' if 'udp.length' in row else 'icmp'
        flow_key = (src, dst, proto)
        
        self.flows[flow_key].append({
            'timestamp': float(row.get('frame.time_epoch', 0)),
            'size': int(row.get('frame.len', 0)),
            'flags': int(row.get('tcp.flags', row.get('icmp.type', 0))),
            'duration': 0
        })
        self._update_graph_entities(src, dst)

    def _finalize_graph(self) -> Optional[Dict]:
        """Final processing steps"""
        self._calculate_flow_stats()
        graph_data = self._build_graph_data()
        self._print_summary()
        return graph_data

    def _process_packet(self, pkt):
        """Original pyshark packet processor (unchanged)"""
        try:
            if not hasattr(pkt, 'ip'):
                return
                
            src = str(pkt.ip.src)
            dst = str(pkt.ip.dst)
            
            proto = next(
                (p for p in ['tcp', 'udp', 'icmp'] if hasattr(pkt, p)),
                None
            )
            
            if proto:
                flow_key = (src, dst, proto)
                features = {
                    'timestamp': float(getattr(pkt, 'sniff_timestamp', 0)),
                    'size': int(getattr(pkt, 'length', 0)),
                    'flags': self._get_flags(pkt),
                    'duration': 0
                }
                self.flows[flow_key].append(features)
                self._update_graph_entities(src, dst)
        except Exception:
            pass

    def _get_flags(self, pkt) -> int:
        """Original flag extractor (unchanged)"""
        try:
            if hasattr(pkt, 'tcp') and hasattr(pkt.tcp, 'flags'):
                return int(pkt.tcp.flags, 16)
            elif hasattr(pkt, 'icmp') and hasattr(pkt.icmp, 'type'):
                return int(pkt.icmp.type)
            return 0
        except:
            return 0

    def _update_graph_entities(self, src: str, dst: str):
        """Original graph updater (unchanged)"""
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
        """Original flow calculator (unchanged)"""
        for flow_key, packets in self.flows.items():
            if len(packets) > 1:
                timestamps = [p['timestamp'] for p in packets if 'timestamp' in p]
                if timestamps:
                    flow_duration = max(timestamps) - min(timestamps)
                    for p in packets:
                        p['duration'] = flow_duration

    def _build_graph_data(self) -> Optional[Dict]:
        """Optimized graph builder with numpy"""
        if not self.node_features:
            print("No valid nodes found in PCAP")
            return None
            
        num_nodes = len(self.node_features)
        features = np.zeros((num_nodes, 13), dtype=np.float32)
        node_names = list(self.node_features.keys())
        
        for i, node in enumerate(node_names):
            relevant = [(f, p) for f in self.flows for p in self.flows[f] if node in f[:2]]
            if not relevant:
                continue
                
            sizes = [p['size'] for _, p in relevant]
            durations = [p['duration'] for _, p in relevant]
            flags = [p['flags'] for _, p in relevant]
            
            features[i] = [
                self.node_features[node]['connections'],
                len({f for f, _ in relevant}),
                1 if self.node_features[node]['type'] == 'host' else 0,
                time.time() - self.node_features[node]['first_seen'],
                len(sizes),
                np.mean(sizes) if sizes else 0,
                np.std(sizes) if sizes else 0,
                np.mean(durations) if durations else 0,
                np.std(durations) if durations else 0,
                len(flags),
                np.mean(flags) if flags else 0,
                np.std(flags) if flags else 0,
                int(any(f > 0 for f in flags))
            ]
        
        if self.edge_connections:
            edge_index = np.array([
                [node_names.index(src), node_names.index(dst)]
                for src, dst in self.edge_connections
            ]).T
            edge_attr = np.array([
                sum(1 for f in self.flows if (src in f[:2] and dst in f[:2]))
                for src, dst in self.edge_connections
            ], dtype=np.float32)
        else:
            edge_index = np.array([[0], [0]], dtype=np.int64)
            edge_attr = np.array([0.0], dtype=np.float32)
        
        return {
            'nodes': node_names,
            'x': torch.tensor(features, dtype=torch.float16 if self.device.type == 'cuda' else torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attr),
            'y': torch.zeros(num_nodes, dtype=torch.long)
        }

    def _print_summary(self):
        """Original summary printer (unchanged)"""
        processing_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"Processing complete")
        print(f"Total packets processed: {self.packet_count}")
        print(f"Total nodes created: {len(self.node_features)}")
        print(f"Total edges created: {len(self.edge_connections)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Processing rate: {self.packet_count/processing_time:.2f} packets/sec")
        print("=" * 60 + "\n")

    def _cleanup_temp_resources(self):
        """Original cleanup (unchanged)"""
        try:
            if hasattr(pyshark.capture.capture, '_current_tshark'):
                del pyshark.capture.capture._current_tshark
            import gc
            gc.collect()
        except:
            pass
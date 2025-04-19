import tempfile
import time
import pyshark
import numpy as np
from collections import defaultdict
import torch
import warnings
from typing import Dict, Optional
import subprocess
import csv
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

class PCAPProcessor:
    def __init__(self):
        print("\n=== INITIALIZING PCAP PROCESSOR ===")
        self.flows = defaultdict(list)
        self.node_features = {}
        self.edge_connections = set()
        self.packet_count = 0
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp()
        print(f"[DEBUG] Temporary directory: {self.temp_dir}")
        
        # Automatic device detection and initialization
        print("[DEBUG] Initializing device...")
        self.device = self._initialize_device()
        print(f"✅ Processor initialized | Device: {self.device}")

    def _initialize_device(self):
        """Handle GPU initialization with fallback to CPU"""
        if torch.cuda.is_available():
            try:
                print("[DEBUG] Testing CUDA device...")
                for i in range(3):
                    x = torch.randn(1024, 1024, device='cuda')
                    torch.mm(x, x.T)
                    print(f"[DEBUG] CUDA test {i+1}/3 passed")
                torch.cuda.synchronize()
                print("[DEBUG] CUDA tests completed successfully")
                return torch.device('cuda')
            except Exception as e:
                print(f"⚠️ GPU initialization failed: {str(e)}")
                return torch.device('cpu')
        print("[DEBUG] Using CPU as fallback")
        return torch.device('cpu')

    def process_pcap(self, pcap_path: str, max_packets: Optional[int] = None) -> Optional[Dict]:
        """Main processing pipeline with debug info"""
        print(f"\n=== PROCESSING PCAP: {os.path.basename(pcap_path)} ===")
        print(f"[DEBUG] Max packets: {max_packets or 'unlimited'}")
        print(f"[DEBUG] Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
        
        try:
            # Try faster tshark CLI first
            try:
                print("[DEBUG] Attempting tshark CLI processing...")
                result = self._process_via_tshark_cli(pcap_path, max_packets)
                print("[DEBUG] tshark processing completed successfully")
            except Exception as e:
                print(f"⚠️ tshark CLI failed, falling back to pyshark: {str(e)}")
                print("[DEBUG] Starting pyshark fallback processing...")
                result = self._process_via_pyshark(pcap_path, max_packets)
                print("[DEBUG] pyshark processing completed")
            
            if result:
                print("[DEBUG] Finalizing graph...")
                final_graph = self._finalize_graph()
                
                if self.device.type == 'cuda':
                    print("[DEBUG] Moving tensors to GPU...")
                    final_graph['x'] = final_graph['x'].pin_memory().to('cuda', non_blocking=True)
                    final_graph['edge_index'] = final_graph['edge_index'].pin_memory().to('cuda', non_blocking=True)
                    final_graph['edge_attr'] = final_graph['edge_attr'].pin_memory().to('cuda', non_blocking=True)
                    final_graph['y'] = final_graph['y'].pin_memory().to('cuda', non_blocking=True)
                    torch.cuda.synchronize()
                    print("[DEBUG] Tensors moved to GPU")
                
                return final_graph
            return None
            
        except Exception as e:
            print(f"❌ PCAP processing failed: {str(e)}")
            return None
        finally:
            print("[DEBUG] Cleaning up temporary resources...")
            self._cleanup_temp_resources()
            print(f"[DEBUG] Total processing time: {time.time() - self.start_time:.2f} seconds")

    def _process_via_pyshark(self, pcap_path: str, max_packets: Optional[int]) -> Optional[Dict]:
        """Fallback to pyshark with debug info"""
        print(f"[DEBUG] Initializing pyshark capture for {pcap_path}")
        cap = pyshark.FileCapture(
            pcap_path,
            display_filter='ip and (tcp or udp or icmp) and !dns',
            keep_packets=False,
            use_json=True,
            include_raw=False
        )
        
        chunk_size = 100000
        processed_in_chunk = 0
        print("[DEBUG] Starting packet processing...")
        
        for pkt in cap:
            if max_packets and self.packet_count >= max_packets:
                print(f"[DEBUG] Reached max packets limit at {self.packet_count}")
                break
                
            self._process_packet(pkt)
            self.packet_count += 1
            processed_in_chunk += 1
            
            if processed_in_chunk >= chunk_size:
                print(f"[DEBUG] Processed {chunk_size} packets, cleaning resources...")
                self._cleanup_temp_resources()
                processed_in_chunk = 0
                
        cap.close()
        print(f"[DEBUG] Processed {self.packet_count} packets total")
        return self._finalize_graph()

    def _process_via_tshark_cli(self, pcap_path: str, max_packets: Optional[int]) -> Optional[Dict]:
        """TSV processing with debug output"""
        print(f"[DEBUG] Starting tshark CLI processing for {pcap_path}")
        csv_path = os.path.join(self.temp_dir, "packets.csv")
        print(f"[DEBUG] Temporary CSV path: {csv_path}")
        
        cmd = [
            'tshark',
            '-r', pcap_path,
            '-o', 'gui.max_tree_depth:200',
            '--disable-protocol', 'mp2t',
            '-T', 'fields',
            '-E', 'header=y',
            '-E', 'separator=,',
            '-E', 'quote=n',
            '-e', 'ip.src',
            '-e', 'ip.dst',
            '-e', 'frame.time_epoch',
            '-e', 'frame.len',
            '-e', 'tcp.flags',
            '-e', 'udp.length',
            '-e', 'icmp.type',
            '-Y', 'ip and (tcp or udp or icmp) and !dns',
            *(['-c', str(max_packets)] if max_packets else [])
        ]

        print(f"[DEBUG] Executing command: {' '.join(cmd)}")
        try:
            with open(csv_path, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)

            file_size = os.path.getsize(csv_path)/1024
            print(f"[DEBUG] CSV file created (size: {file_size:.2f} KB)")
            
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                print(f"[DEBUG] CSV headers: {reader.fieldnames}")
                
                for i, row in enumerate(reader):
                    if max_packets and self.packet_count >= max_packets:
                        print(f"[DEBUG] Reached max packets limit at {self.packet_count}")
                        break
                    
                    if i % 100000 == 0:
                        print(f"[DEBUG] Processing packet {i}...")
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    self._process_csv_row(row)
                    self.packet_count += 1
            
            print(f"[DEBUG] Processed {self.packet_count} packets total")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Tshark failed with return code {e.returncode}")
            raise Exception(f"Tshark failed with error {e.returncode}")

    def _process_csv_row(self, row: Dict):
        """Row processing with detailed debug"""
        try:
            if not isinstance(row, dict):
                print(f"[WARN] Row is not a dictionary: {type(row)}")
                return

            src = str(row.get('ip.src', '')).strip('"\'\\ ')
            dst = str(row.get('ip.dst', '')).strip('"\'\\ ')
            
            if not src or not dst:
                print(f"[WARN] Missing IPs in row: src={src}, dst={dst}")
                return

            # Debug print first packet and every 100,000th packet
            if self.packet_count == 0 or self.packet_count % 100000 == 0:
                print(f"[DEBUG] Sample packet {self.packet_count}:")
                print(f"  src: {src}, dst: {dst}")
                print(f"  timestamp: {row.get('frame.time_epoch')}")
                print(f"  size: {row.get('frame.len')}")
                print(f"  flags: {row.get('tcp.flags')}")

            # Safely parse numeric fields
            try:
                timestamp_str = str(row.get('frame.time_epoch', '0')).strip('"\'\\ ')
                timestamp = float(timestamp_str) if timestamp_str.replace('.', '', 1).isdigit() else 0.0
            except (ValueError, TypeError) as e:
                print(f"⚠️ Timestamp conversion error: {str(e)}")
                timestamp = 0.0

            try:
                size_str = str(row.get('frame.len', '0')).strip('"\'\\ ')
                size = int(float(size_str)) if size_str.replace('.', '', 1).isdigit() else 0
            except (ValueError, TypeError) as e:
                print(f"⚠️ Size conversion error: {str(e)}")
                size = 0

            # Protocol and flag detection
            proto = 'unknown'
            flags = 0
            
            # Check TCP flags first
            tcp_flags = str(row.get('tcp.flags', '')).strip('"\'\\ ')
            if tcp_flags:
                try:
                    flags = int(tcp_flags, 16) if tcp_flags.startswith('0x') else int(tcp_flags)
                    proto = 'tcp'
                except (ValueError, TypeError):
                    pass
            # Fallback to ICMP type
            elif 'icmp.type' in row:
                try:
                    flags = int(str(row['icmp.type']).strip('"\'\\ '))
                    proto = 'icmp'
                except (ValueError, TypeError):
                    pass
            # Final fallback to UDP
            elif 'udp.length' in row:
                proto = 'udp'

            # Store the flow data
            flow_key = (src, dst, proto)
            self.flows[flow_key].append({
                'timestamp': timestamp,
                'size': size,
                'flags': flags,
                'duration': 0
            })
            self._update_graph_entities(src, dst)
            
        except Exception as e:
            print(f"⚠️ Error processing packet: {str(e)}")
            if 'row' in locals():
                print(f"Problematic row data: {dict((k,v) for k,v in row.items() if v)}")

    def _finalize_graph(self) -> Optional[Dict]:
        """Graph finalization with debug info"""
        print("\n[DEBUG] Finalizing graph data...")
        print(f"[DEBUG] Current stats - Flows: {len(self.flows)}, Nodes: {len(self.node_features)}, Edges: {len(self.edge_connections)}")
        
        self._calculate_flow_stats()
        
        if not self.node_features:
            print("[ERROR] No valid nodes found in PCAP")
            return None
            
        num_nodes = len(self.node_features)
        print(f"[DEBUG] Creating feature matrix for {num_nodes} nodes")
        
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
        
            if i < 3:  # Debug print for first 3 nodes
                print(f"[DEBUG] Sample node {i}: {node}")
                print(f"  connections: {features[i][0]}")
                print(f"  mean size: {features[i][5]:.2f}")
                print(f"  mean duration: {features[i][7]:.2f}")

        # Build edge connections
        if self.edge_connections:
            print(f"[DEBUG] Creating {len(self.edge_connections)} edges")
            edge_index = np.array([
                [node_names.index(src), node_names.index(dst)]
                for src, dst in self.edge_connections
            ]).T
            edge_attr = np.array([
                sum(1 for f in self.flows if (src in f[:2] and dst in f[:2]))
                for src, dst in self.edge_connections
            ], dtype=np.float32)
        else:
            print("[DEBUG] No edges found, creating placeholder")
            edge_index = np.array([[0], [0]], dtype=np.int64)
            edge_attr = np.array([0.0], dtype=np.float32)
        
        # Use mixed precision for GPU
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        print(f"[DEBUG] Using dtype: {dtype}")
        
        graph_data = {
            'nodes': node_names,
            'x': torch.as_tensor(features, dtype=dtype),
            'edge_index': torch.as_tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.as_tensor(edge_attr, dtype=dtype),
            'y': torch.zeros(num_nodes, dtype=torch.long)
        }
        
        self._print_summary()
        return graph_data

    def _process_packet(self, pkt):
        """Original pyshark packet processor with debug"""
        try:
            if not hasattr(pkt, 'ip'):
                print("[DEBUG] Skipping non-IP packet")
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
        except Exception as e:
            print(f"[DEBUG] Error processing packet: {str(e)}")

    def _get_flags(self, pkt) -> int:
        """Flag extractor with debug"""
        try:
            if hasattr(pkt, 'tcp') and hasattr(pkt.tcp, 'flags'):
                return int(pkt.tcp.flags, 16)
            elif hasattr(pkt, 'icmp') and hasattr(pkt.icmp, 'type'):
                return int(pkt.icmp.type)
            return 0
        except Exception as e:
            print(f"[DEBUG] Flag extraction error: {str(e)}")
            return 0

    def _update_graph_entities(self, src: str, dst: str):
        """Graph updater with debug"""
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
        """Flow calculator with debug"""
        print("[DEBUG] Calculating flow statistics...")
        for flow_key, packets in self.flows.items():
            if len(packets) > 1:
                timestamps = [p['timestamp'] for p in packets if 'timestamp' in p]
                if timestamps:
                    flow_duration = max(timestamps) - min(timestamps)
                    for p in packets:
                        p['duration'] = flow_duration
        print("[DEBUG] Flow stats calculated")

    def _print_summary(self):
        """Enhanced summary with GPU info"""
        processing_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"📊 Processing complete")
        print(f"📦 Total packets processed: {self.packet_count}")
        print(f"🖥️ Total nodes created: {len(self.node_features)}")
        print(f"🔗 Total edges created: {len(self.edge_connections)}")
        
        if self.device.type == 'cuda':
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory Used: {alloc:.2f}GB / {reserved:.2f}GB")
        
        print(f"⏱ Processing time: {processing_time:.2f} seconds")
        print(f"🚀 Processing rate: {self.packet_count/processing_time:.2f} packets/sec")
        print("=" * 60 + "\n")

    def _cleanup_temp_resources(self):
        """Enhanced cleanup with GPU memory management"""
        try:
            if hasattr(pyshark.capture.capture, '_current_tshark'):
                del pyshark.capture.capture._current_tshark
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[DEBUG] GPU memory cleanup | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
                
            import gc
            gc.collect()
        except Exception as e:
            print(f"[DEBUG] Cleanup error: {str(e)}")
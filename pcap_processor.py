import tempfile
import time
import pandas as pd
import pyshark
import numpy as np
from collections import defaultdict
import torch
import warnings
from typing import Dict, Optional
import subprocess
import csv
import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

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

            # Print feature statistics for the first 5 nodes
            print("\n[DEBUG] Feature Verification:")
            for i, (node, feats) in enumerate(list(self.node_features.items())[:5]):
                print(f"Node {i}: {node}")
                print(f"  Connections: {feats['connections']}")
                print(f"  First seen: {feats['first_seen']}")
                print(f"  Type: {feats['type']}")

            # Print sample flow stats
            sample_flow = next(iter(self.flows.values()))
            print("\n[DEBUG] Sample Flow Stats:")
            print(f"  Packet count: {len(sample_flow)}")
            print(f"  Avg size: {np.mean([p['size'] for p in sample_flow]):.2f}")
            print(f"  Avg duration: {np.mean([p['duration'] for p in sample_flow]):.2f}")
            
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
        """Ultimate robust tshark processing with all fixes"""
        print(f"[DEBUG] Starting ultra-robust tshark processing for {pcap_path}")
        csv_path = os.path.join(self.temp_dir, "packets.tsv")
        
        # Tshark command with all known fixes
        cmd = [
            'tshark',
            '-r', pcap_path,
            '-o', 'gui.max_tree_depth:200',
            '--disable-protocol', 'mp2t',  # Only disable protocols that definitely exist
            '-T', 'fields',
            '-E', 'header=y',
            '-E', 'separator=\t',
            '-E', 'occurrence=f',
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

        print(f"[DEBUG] Executing: {' '.join(cmd)}")
        
        try:
            # Run with full error capture
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Write output to file
            with open(csv_path, 'w') as f:
                f.write(result.stdout)
            
            # Process output line by line
            line_count = 0
            reader = csv.DictReader(
                result.stdout.splitlines(),
                delimiter='\t',
                fieldnames=[
                    'ip.src', 'ip.dst', 'frame.time_epoch',
                    'frame.len', 'tcp.flags', 'udp.length', 'icmp.type'
                ]
            )
            
            for row in reader:
                if max_packets and self.packet_count >= max_packets:
                    break
                    
                try:
                    self._process_csv_row(row)
                    self.packet_count += 1
                    line_count += 1
                    
                    if line_count % 100000 == 0:
                        print(f"[DEBUG] Processed {line_count} packets...")
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"⚠️ Packet {line_count} error: {str(e)}")
                    continue
                    
            print(f"[DEBUG] Successfully processed {self.packet_count} packets")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Tshark failed (code {e.returncode})")
            print(f"Stderr: {e.stderr[:500]}...")  # Print first 500 chars of error
            raise Exception("Tshark processing failed")

    def _process_dask_partition(self, df_partition):
        """Helper for Dask parallel processing"""
        count = 0
        for _, row in df_partition.iterrows():
            self._process_csv_row(row.to_dict())
            count += 1
        return pd.DataFrame({'count': [count]})

    def _process_csv_row(self, row: Dict):
        """Row processing with detailed debug"""
        try:

            if row.get('ip.src') == 'ip.src' or row.get('ip.dst') == 'ip.dst':
                return

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
        """Complete updated graph finalization with robust feature engineering"""
        print("\n[DEBUG] Finalizing graph data...")
        try:
            # --- Node Feature Construction ---
            node_names = list(self.node_features.keys())
            features = np.zeros((len(node_names), 13), dtype=np.float32)
            
            # Calculate global stats for normalization
            all_connections = [n['connections'] for n in self.node_features.values()]
            global_conn_mean = np.mean(all_connections) if all_connections else 1
            global_conn_std = np.std(all_connections) if all_connections else 1

            for i, (node, node_data) in enumerate(self.node_features.items()):
                # Get all flows for this node (safe access)
                relevant_flows = []
                for flow_key, packets in self.flows.items():
                    if node in flow_key[:2]:
                        sizes = [p.get('size', 0) for p in packets]
                        durations = [p.get('duration', 0) for p in packets]
                        flags = [p.get('flags', 0) for p in packets]
                        relevant_flows.append({
                            'sizes': sizes,
                            'durations': durations,
                            'flags': flags,
                            'count': len(packets)
                        })

                # Concatenate all values with empty array fallback
                sizes = np.concatenate([f['sizes'] for f in relevant_flows]) if relevant_flows else np.array([0])
                durations = np.concatenate([f['durations'] for f in relevant_flows]) if relevant_flows else np.array([0])
                flags = np.concatenate([f['flags'] for f in relevant_flows]) if relevant_flows else np.array([0])

                # --- New Feature Engineering ---
                features[i] = [
                    # 1. Log-normalized connection count
                    np.log1p(node_data['connections']) / 15.0,
                    
                    # 2. Log flow count
                    np.log1p(len(relevant_flows)) / 10.0,
                    
                    # 3. Binary host indicator (unchanged)
                    1.0 if node_data.get('type') == 'host' else 0.0,
                    
                    # 4. Hours since first seen (sigmoid normalized)
                    1 / (1 + np.exp(-(time.time() - node_data.get('first_seen', time.time())) / 3600)),
                    
                    # 5-7. Packet size stats
                    np.log1p(np.mean(sizes)) / 15.0,
                    np.log1p(np.std(sizes)) / 10.0 if len(sizes) > 1 else 0.0,
                    np.mean(sizes > 1500),  # Jumbo frame ratio
                    
                    # 8-10. Duration stats
                    np.log1p(np.mean(durations)) / 10.0,
                    np.log1p(np.std(durations)) / 10.0 if len(durations) > 1 else 0.0,
                    np.mean(durations > 1),  # Long connection ratio
                    
                    # 11-13. Flag stats
                    np.log1p(np.sum(flags)) / 10.0,
                    np.log1p(np.std(flags)) / 5.0 if len(flags) > 1 else 0.0,
                    min(np.sum([f in {0x12, 0x29, 3, 8} for f in flags]) / 10.0, 1.0)
                ]

            # --- Edge Construction (unchanged from previous) ---
            edge_index = []
            edge_attr = []
            node_to_idx = {name: idx for idx, name in enumerate(node_names)}
            
            for src, dst in self.edge_connections:
                try:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edge_index.append([src_idx, dst_idx])
                    flow_count = sum(1 for flow in self.flows if src in flow[:2] and dst in flow[:2])
                    edge_attr.append(np.log1p(flow_count))
                except KeyError:
                    continue

            # Convert to tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            
            # Final validation and clipping
            features = np.nan_to_num(features, nan=0.0)
            features = np.clip(features, 0.0, 1.0)  # Hard clip to [0,1] range
            
            print("\n[DEBUG] Final Feature Statistics:")
            print(f"Min: {np.min(features):.4f}, Max: {np.max(features):.4f}")
            print(f"Mean: {np.mean(features):.4f}")
            print("Sample Node Features:", features[0])

            return {
                'nodes': node_names,
                'x': torch.as_tensor(features, dtype=torch.float32),
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'y': torch.zeros(len(node_names), dtype=torch.long)
            }

        except Exception as e:
            print(f"❌ Graph finalization failed: {str(e)}")
            return None
    
    def _safe_numeric_conversion(self, value, default=0.0):
        """Robust numeric conversion with error handling"""
        try:
            if isinstance(value, str):
                value = value.strip('"\'\\ ')
                if value.lower() in ('', 'nan', 'inf', '-inf'):
                    return default
            return float(value)
        except (ValueError, TypeError):
            return default

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
        for flow_key, packets in self.flows.items():
            try:
                if len(packets) > 1:
                    timestamps = [p.get('timestamp', 0) for p in packets]
                    if timestamps:
                        flow_duration = max(timestamps) - min(timestamps)
                        for p in packets:
                            p['duration'] = flow_duration if flow_duration >= 0 else 0
            except:
                continue

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
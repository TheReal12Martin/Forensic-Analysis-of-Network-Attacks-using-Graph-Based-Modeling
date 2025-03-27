import os
import dpkt
import socket
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
from collections import defaultdict
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Final 43 features that MUST match the model
FINAL_FEATURES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
    'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'label'
]

class ConnectionTracker:
    def __init__(self):
        self.connections = defaultdict(self._create_connection)
        self.flow_stats = defaultdict(self._create_flow_stats)
    
    @staticmethod
    def _create_connection():
        return {
            'src_pkts': 0, 'dst_pkts': 0, 'src_bytes': 0, 'dst_bytes': 0,
            'start_time': None, 'last_time': None, 'intervals': []
        }
    
    @staticmethod
    def _create_flow_stats():
        return {
            'ct_srv_src': 0, 'ct_state_ttl': 0, 'ct_dst_ltm': 0,
            'ct_src_dport_ltm': 0, 'ct_dst_sport_ltm': 0, 'ct_dst_src_ltm': 0,
            'ct_ftp_cmd': 0, 'ct_flw_http_mthd': 0, 'ct_src_ltm': 0,
            'ct_srv_dst': 0
        }

def ip_to_numeric(ip):
    """Convert IP address to consistent numeric value"""
    if isinstance(ip, bytes):
        try:
            ip_str = socket.inet_ntoa(ip)
        except:
            ip_str = str(ip)
    else:
        ip_str = str(ip)
    return int(hashlib.md5(ip_str.encode('utf-8')).hexdigest()[:8], 16) % (2**32)

def get_protocol_name(proto_num):
    protocol_map = {
        1: 'icmp', 6: 'tcp', 17: 'udp', 2: 'igmp',
        47: 'gre', 50: 'esp', 51: 'ah', 58: 'icmp6'
    }
    return protocol_map.get(proto_num, f'unknown({proto_num})')

def get_service_name(dport, proto):
    service_map = {
        'tcp': {80: 'http', 443: 'ssl', 22: 'ssh', 21: 'ftp'},
        'udp': {53: 'dns', 123: 'ntp'}
    }
    return service_map.get(proto, {}).get(dport, '-')

def get_connection_state(tcp_flags):
    if not tcp_flags: return 'OTH'
    if tcp_flags & dpkt.tcp.TH_SYN and tcp_flags & dpkt.tcp.TH_ACK: return 'S2'
    if tcp_flags & dpkt.tcp.TH_ACK: return 'S1'
    return 'REQ'

def update_flow_stats(conn_tracker, src_ip, dst_ip, sport, dport, proto, service):
    conn_tracker.flow_stats[(src_ip, service)]['ct_srv_src'] += 1
    conn_tracker.flow_stats[(src_ip, dst_ip)]['ct_dst_ltm'] += 1
    conn_tracker.flow_stats[(src_ip, dport)]['ct_src_dport_ltm'] += 1
    conn_tracker.flow_stats[(dst_ip, sport)]['ct_dst_sport_ltm'] += 1
    conn_tracker.flow_stats[(dst_ip, src_ip)]['ct_dst_src_ltm'] += 1
    conn_tracker.flow_stats[src_ip]['ct_src_ltm'] += 1
    conn_tracker.flow_stats[(service, dst_ip)]['ct_srv_dst'] += 1
    conn_tracker.flow_stats[(src_ip, dst_ip)]['ct_state_ttl'] += 1

def initialize_features():
    return {name: 0 for name in FINAL_FEATURES}

def process_packet(ts, pkt, conn_tracker):
    features = initialize_features()
    
    try:
        eth = dpkt.ethernet.Ethernet(pkt)
        ip = eth.data if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)) else None
        if not ip:
            return features

        # Handle both IPv4 and IPv6
        src_ip = ip.src if isinstance(ip, dpkt.ip.IP) else ip.src
        dst_ip = ip.dst if isinstance(ip, dpkt.ip.IP) else ip.dst
        proto_num = ip.p
        proto = get_protocol_name(proto_num)
        
        transport = ip.data
        sport = dport = 0
        tcp_flags = 0
        if isinstance(transport, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            sport, dport = transport.sport, transport.dport
            if isinstance(transport, dpkt.tcp.TCP):
                tcp_flags = transport.flags
        
        # Connection tracking
        conn_key = (src_ip, sport, dst_ip, dport, proto)
        conn = conn_tracker.connections[conn_key]
        
        current_time = float(ts)
        if conn['start_time'] is None:
            conn['start_time'] = current_time
            duration = 0.0
        else:
            duration = current_time - conn['start_time']
        conn['last_time'] = current_time
        
        # Packet processing
        pkt_size = len(transport) if transport else 0
        conn['src_pkts'] += 1
        conn['src_bytes'] += pkt_size
        
        # Feature calculation
        service = get_service_name(dport, proto)
        state = get_connection_state(tcp_flags)
        update_flow_stats(conn_tracker, src_ip, dst_ip, sport, dport, proto, service)
        
        intervals = conn['intervals']
        mean_interval = np.mean(intervals) if intervals else 0
        sjit = np.std(intervals) if len(intervals) > 1 else 0
        sload = pkt_size / duration if duration > 0 else 0
        
        # Populate all 43 features
        features.update({
            'srcip': ip_to_numeric(src_ip),
            'sport': sport,
            'dstip': ip_to_numeric(dst_ip),
            'dsport': dport,
            'proto': proto_num,
            'state': 1 if state == 'S1' else (2 if state == 'S2' else 0),
            'dur': duration,
            'sbytes': pkt_size,
            'dbytes': 0,
            'sttl': ip.ttl if hasattr(ip, 'ttl') else 64,
            'dttl': ip.ttl if hasattr(ip, 'ttl') else 64,
            'sloss': 0,
            'dloss': 0,
            'service': 1 if service == 'http' else (2 if service == 'ssl' else 0),
            'sload': sload,
            'dload': 0,
            'sinpkt': mean_interval,
            'dinpkt': 0,
            'sjit': sjit,
            'djit': 0,
            'swin': transport.window if hasattr(transport, 'window') else 0,
            'dwin': 0,
            'stcpb': transport.seq if hasattr(transport, 'seq') else 0,
            'dtcpb': 0,
            'smeansz': conn['src_bytes'] / conn['src_pkts'] if conn['src_pkts'] > 0 else 0,
            'dmeansz': 0,
            'trans_depth': 1 if tcp_flags & dpkt.tcp.TH_SYN else 0,
            'response_body_len': pkt_size,
            'ct_srv_src': conn_tracker.flow_stats[(src_ip, service)]['ct_srv_src'],
            'ct_state_ttl': conn_tracker.flow_stats[(src_ip, dst_ip)]['ct_state_ttl'],
            'ct_dst_ltm': conn_tracker.flow_stats[(src_ip, dst_ip)]['ct_dst_ltm'],
            'ct_src_dport_ltm': conn_tracker.flow_stats[(src_ip, dport)]['ct_src_dport_ltm'],
            'ct_dst_sport_ltm': conn_tracker.flow_stats[(dst_ip, sport)]['ct_dst_sport_ltm'],
            'ct_dst_src_ltm': conn_tracker.flow_stats[(dst_ip, src_ip)]['ct_dst_src_ltm'],
            'is_ftp_login': 1 if service == 'ftp' and pkt_size > 0 else 0,
            'ct_ftp_cmd': conn_tracker.flow_stats[(src_ip, 'ftp')]['ct_ftp_cmd'],
            'ct_flw_http_mthd': conn_tracker.flow_stats[(src_ip, 'http')]['ct_flw_http_mthd'],
            'ct_src_ltm': conn_tracker.flow_stats[src_ip]['ct_src_ltm'],
            'ct_srv_dst': conn_tracker.flow_stats[(service, dst_ip)]['ct_srv_dst'],
            'is_sm_ips_ports': 1 if src_ip == dst_ip and sport == dport else 0,
            'label': 0
        })
        
    except Exception as e:
        logger.error(f"Error processing packet: {e}")
        return initialize_features()
    
    return features

def extract_pcap(pcap_file, output_csv):
    if not os.path.exists(pcap_file):
        logger.error(f"Input file not found: {pcap_file}")
        return
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    conn_tracker = ConnectionTracker()
    batch_data = []
    
    with open(pcap_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        total_packets = sum(1 for _ in pcap)
    
    with open(pcap_file, 'rb') as f, open(output_csv, 'w') as out_file:
        pcap = dpkt.pcap.Reader(f)
        writer = None
        
        for ts, buf in tqdm(pcap, total=total_packets, desc="Processing packets"):
            features = process_packet(ts, buf, conn_tracker)
            batch_data.append(features)
            
            if len(batch_data) >= 50000:
                df = pd.DataFrame(batch_data)
                # Ensure we have exactly 43 features
                missing = set(FINAL_FEATURES) - set(df.columns)
                if missing:
                    logger.error(f"Missing features in batch: {missing}")
                    for feat in missing:
                        df[feat] = 0
                df = df[FINAL_FEATURES]
                
                if writer is None:
                    df.to_csv(out_file, index=False)
                    writer = True
                else:
                    df.to_csv(out_file, mode='a', header=False, index=False)
                batch_data = []
                gc.collect()
        
        if batch_data:
            df = pd.DataFrame(batch_data)
            missing = set(FINAL_FEATURES) - set(df.columns)
            if missing:
                logger.error(f"Missing features in final batch: {missing}")
                for feat in missing:
                    df[feat] = 0
            df = df[FINAL_FEATURES]
            df.to_csv(out_file, mode='a', header=False, index=False)
    
    # Final verification
    df_check = pd.read_csv(output_csv)
    if len(df_check.columns) != len(FINAL_FEATURES):
        logger.error(f"Feature count mismatch. Expected {len(FINAL_FEATURES)}, got {len(df_check.columns)}")
        raise ValueError("Output CSV has incorrect number of features")
    logger.info(f"Successfully extracted {len(df_check)} records with {len(df_check.columns)} features")

def extract_features():
    pcap_file = "/home/martin/Original Network Traffic and Log data/Friday-02-03-2018/pcap/capPC1-172.31.65.77"
    output_csv = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018.csv"
    
    logger.info("Starting feature extraction...")
    extract_pcap(pcap_file, output_csv)
    logger.info(f"Feature extraction complete. Results saved to {output_csv}")

if __name__ == "__main__":
    extract_features()
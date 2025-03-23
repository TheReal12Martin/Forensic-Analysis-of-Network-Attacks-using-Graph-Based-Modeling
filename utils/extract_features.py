import os
from scapy.all import PcapReader, IP, TCP, UDP, IPv6, Raw
import pandas as pd
from tqdm import tqdm  # For progress bar

def extract_pcap_features(pcap_file, output_csv):
    """Extract features from a single PCAP file."""
    # Use PcapReader for streaming packets
    packets = PcapReader(pcap_file)
    data = []
    last_packet_time = None  # Track the last packet time for duration calculation

    # Iterate over packets with a progress bar
    for i, pkt in tqdm(enumerate(packets), desc="Processing packets"):
        # Check if the packet has an IP layer (IPv4 or IPv6)
        if IP in pkt or IPv6 in pkt:
            row = {
                'srcip': pkt[IP].src if IP in pkt else pkt[IPv6].src,
                'sport': pkt.sport if TCP in pkt or UDP in pkt else 0,
                'dstip': pkt[IP].dst if IP in pkt else pkt[IPv6].dst,
                'dsport': pkt.dport if TCP in pkt or UDP in pkt else 0,
                'proto': pkt[IP].proto if IP in pkt else pkt[IPv6].nh,
                'state': 'ESTABLISHED' if TCP in pkt and pkt[TCP].flags == 0x18 else 'OTHER',
                'dur': pkt.time if i == 0 else pkt.time - last_packet_time,
                'sbytes': len(pkt[IP].payload) if IP in pkt and Raw in pkt else len(pkt[IPv6].payload) if IPv6 in pkt and Raw in pkt else 0,
                'dbytes': 0,  # Placeholder for destination bytes
                'sttl': pkt[IP].ttl if IP in pkt else pkt[IPv6].hlim,
                'dttl': pkt[IP].ttl if IP in pkt else pkt[IPv6].hlim,
                'sloss': 0,  # Placeholder for source packet loss
                'dloss': 0,  # Placeholder for destination packet loss
                'service': 'http' if TCP in pkt and pkt[TCP].dport == 80 else '-',
                'sload': 0,  # Placeholder for source load
                'dload': 0,  # Placeholder for destination load,
                'sinpkt': 0 if i == 0 else pkt.time - last_packet_time,
                'dinpkt': 0,  # Placeholder for destination inter-packet arrival time
                'sjit': 0,  # Placeholder for source jitter
                'djit': 0,  # Placeholder for destination jitter
                'swin': pkt[TCP].window if TCP in pkt else 0,
                'stcpb': pkt[TCP].seq if TCP in pkt else 0,
                'dtcpb': pkt[TCP].seq if TCP in pkt else 0,
                'dwin': pkt[TCP].window if TCP in pkt else 0,
                'tcprtt': 0,  # Placeholder for TCP RTT
                'synack': 0,  # Placeholder for SYN-ACK time
                'ackdat': 0,  # Placeholder for ACK data time
                'smean': 0,  # Placeholder for source mean packet size
                'dmean': 0,  # Placeholder for destination mean packet size
                'trans_depth': 0,  # Placeholder for transaction depth
                'response_body_len': 0,  # Placeholder for response body length
                'ct_srv_src': 0,  # Placeholder for connection count from source to service
                'ct_state_ttl': 0,  # Placeholder for connection state and TTL
                'ct_dst_ltm': 0,  # Placeholder for connection count to destination in last time window
                'ct_src_dport_ltm': 0,  # Placeholder for connection count from source to destination port in last time window
                'ct_dst_sport_ltm': 0,  # Placeholder for connection count from destination to source port in last time window
                'ct_dst_src_ltm': 0,  # Placeholder for connection count from destination to source in last time window
                'is_ftp_login': 0,  # Placeholder for FTP login
                'ct_ftp_cmd': 0,  # Placeholder for FTP command count
                'ct_flw_http_mthd': 0,  # Placeholder for HTTP method count
                'ct_src_ltm': 0,  # Placeholder for connection count from source in last time window
                'ct_srv_dst': 0,  # Placeholder for connection count to service from destination
                'is_sm_ips_ports': 1 if IP in pkt and pkt[IP].src == pkt[IP].dst and pkt.sport == pkt.dport else 0,
                'label': 0  # Placeholder for binary label
            }
            data.append(row)
            last_packet_time = pkt.time  # Update the last packet time

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Features extracted and saved to {output_csv}")


def extract_features():
    # Path to the input PCAP file
    pcap_file = "/home/martin/Original Network Traffic and Log data/Friday-02-03-2018/pcap/capPC1-172.31.65.77"  # Replace with your PCAP file path

    # Path to the output CSV file
    output_csv = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018.csv"  # Replace with your desired output path

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Extract features from the PCAP file
    extract_pcap_features(pcap_file, output_csv)
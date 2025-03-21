import os
from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
import pandas as pd
from tqdm import tqdm  # For progress bar

def extract_pcap_features(pcap_file, output_csv):
    """Extract features from a single PCAP file and save to a CSV."""
    # Read the PCAP file
    packets = rdpcap(pcap_file)
    data = []

    # Process each packet
    for pkt in tqdm(packets, desc="Processing packets"):
        if IP in pkt or IPv6 in pkt:
            # Extract basic features
            row = {
                'srcip': pkt[IP].src if IP in pkt else pkt[IPv6].src,
                'sport': pkt.sport if TCP in pkt or UDP in pkt else 0,
                'dstip': pkt[IP].dst if IP in pkt else pkt[IPv6].dst,
                'dsport': pkt.dport if TCP in pkt or UDP in pkt else 0,
                'proto': pkt[IP].proto if IP in pkt else pkt[IPv6].nh,
                'state': 'INT' if TCP in pkt else 'CON',
                'dur': pkt.time,
                'sbytes': len(pkt[IP].payload) if IP in pkt and Raw in pkt else len(pkt[IPv6].payload) if IPv6 in pkt and Raw in pkt else 0,
                'dbytes': 0,
                'sttl': pkt[IP].ttl if IP in pkt else pkt[IPv6].hlim,
                'dttl': 0,
                'sloss': 0,
                'dloss': 0,
                'service': '-',
                'sload': 0,
                'dload': 0,
                'sinpkt': 0,
                'dinpkt': 0,
                'label': 0,
                'attack_cat': 'Normal'
            }
            data.append(row)

    # Convert the list of features to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Features extracted and saved to {output_csv}")

def extract_features():
    # Path to the input PCAP file
    pcap_file = "/home/martin/Original Network Traffic and Log data/Friday-02-03-2018/pcap/capDESKTOP-AN3U28N-172.31.64.17"  # Replace with your PCAP file path

    # Path to the output CSV file
    output_csv = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018.csv"  # Replace with your desired output path

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Extract features from the PCAP file
    extract_pcap_features(pcap_file, output_csv)
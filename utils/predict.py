import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from models.GAT import GAT
from config import Config
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FEATURES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
    'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'label'
]

def create_ip_graph(df):
    """Create graph with IP-only nodes while preserving all features"""
    G = nx.Graph()
    ip_to_idx = {}
    ip_features = defaultdict(list)
    
    # Verify features without logging empty sets
    missing_features = set(MODEL_FEATURES) - set(df.columns)
    if missing_features:
        logger.warning(f"Adding {len(missing_features)} missing features")
        for feat in missing_features:
            df[feat] = 0
    
    # Ensure correct feature order
    df = df[MODEL_FEATURES]
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building IP graph"):
        src_ip = row['srcip']
        dst_ip = row['dstip']
        
        if pd.isna(src_ip) or pd.isna(dst_ip):
            continue
            
        for ip in [src_ip, dst_ip]:
            ip_features[ip].append(row.values)
            if ip not in ip_to_idx:
                ip_to_idx[ip] = len(ip_to_idx)
        
        src_idx = ip_to_idx[src_ip]
        dst_idx = ip_to_idx[dst_ip]
        if G.has_edge(src_idx, dst_idx):
            G[src_idx][dst_idx]['weight'] += 1
        else:
            G.add_edge(src_idx, dst_idx, weight=1)
    
    # Create feature matrix
    node_features = np.zeros((len(ip_to_idx), len(MODEL_FEATURES)))
    for ip, idx in ip_to_idx.items():
        if ip_features[ip]:
            node_features[idx] = np.mean(ip_features[ip], axis=0)
    
    logger.info(f"Created graph with {len(ip_to_idx)} IP nodes and {len(G.edges())} edges")
    return G, node_features, ip_to_idx

def predict_on_csv(input_csv, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} records")
        
        # Create IP-based graph
        graph, node_features, ip_to_idx = create_ip_graph(df)
        
        # Convert to PyG format
        edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float32)
        data = Data(x=x.to(device), edge_index=edge_index.to(device))
        
        # Initialize model
        model = GAT(
            num_features=len(MODEL_FEATURES),
            hidden_channels=Config.HIDDEN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            heads=Config.HEADS
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Predict
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1).cpu().numpy()
        
        # Save results
        results = pd.DataFrame({
            'ip': list(ip_to_idx.keys()),
            'prediction': preds
        })
        output_csv = input_csv.replace('.csv', '_predictions.csv')
        results.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to {output_csv}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    predict_on_csv("/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018.csv", 
                  "gat_network_security_model.pth")
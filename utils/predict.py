import pandas as pd
import torch
from torch_geometric.data import Data
from models.GAT import GAT
from config import Config

def process_csv(input_csv):
    """Process the CSV file to match the UNSW dataset format."""
    # Load the CSV file
    df = pd.read_csv(input_csv)
    print(f"Number of features in input CSV: {len(df.columns)}")

    # List of 43 features in the UNSW dataset
    unsw_features = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'sinpkt', 'dinpkt',
        'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
        'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
        'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'label'
    ]

    # Add missing features with default values (e.g., 0)
    for feature in unsw_features:
        if feature not in df.columns:
            print(f"Adding missing feature: {feature}")
            df[feature] = 0

    # Select only the 43 features present in the UNSW dataset
    df = df[unsw_features]
    print(f"Number of features after selection: {len(df.columns)}")

    # Fill missing values with default values
    df.fillna(0, inplace=True)

    return df

def create_graph(df):
    """Create a graph from the processed CSV data."""
    import networkx as nx

    # Create a graph
    G = nx.Graph()

    # Add nodes and edges based on communication
    for _, row in df.iterrows():
        src_ip = row['srcip']
        dst_ip = row['dstip']

        # Add source IP as a node if not already added
        if src_ip not in G:
            G.add_node(src_ip)

        # Add destination IP as a node if not already added
        if dst_ip not in G:
            G.add_node(dst_ip)

        # Add an edge between source and destination IPs
        G.add_edge(src_ip, dst_ip)

    return G



def convert_to_pyg_format(graph, df):
    """Convert the graph to PyTorch Geometric format."""
    # Create a mapping from IP addresses to node indices
    ip_to_node = {ip: i for i, ip in enumerate(graph.nodes())}

    # Create edge indices
    edge_index = []
    for src_ip, dst_ip in graph.edges():
        src_node = ip_to_node[src_ip]
        dst_node = ip_to_node[dst_ip]
        edge_index.append([src_node, dst_node])

    # Convert to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Encode categorical features
    df_encoded = df.copy()

    # Encode IP addresses
    df_encoded['srcip'] = df_encoded['srcip'].map(ip_to_node)
    df_encoded['dstip'] = df_encoded['dstip'].map(ip_to_node)

    # Encode other categorical features (e.g., proto, state)
    categorical_cols = ['proto', 'state', 'service']
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # Drop the 'label' column (if present)
    if 'label' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['label'])

    # Convert the DataFrame to a tensor
    x = torch.tensor(df_encoded.values, dtype=torch.float)

    # Debug: Print the number of features
    print(f"Number of features in PyG format: {x.size(1)}")

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    return data



def predict_on_csv(input_csv, model_path):
    """Load the pre-trained GAT model and make predictions on the CSV data."""
    # Step 1: Process the CSV
    print("Processing the CSV file...")
    df = process_csv(input_csv)

    # Step 2: Create a graph
    print("Creating the graph...")
    graph = create_graph(df)

    # Step 3: Convert the graph to PyG format
    print("Converting the graph to PyG format...")
    data = convert_to_pyg_format(graph, df)

    # Debug: Print the number of features
    print(f"Number of features during prediction: {data.num_features}")

    # Step 4: Load the pre-trained GAT model
    print("Loading the pre-trained GAT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_features=data.num_features, hidden_channels=Config.HIDDEN_CHANNELS, num_classes=Config.NUM_CLASSES, heads=Config.HEADS).to(device)
    
    # Load the model state dict
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Ensure the number of features matches the trained model.")
        return

    model.eval()
    data = data.to(device)

    # Step 5: Perform predictions
    print("Making predictions...")
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1).cpu().numpy()

    # Add predictions to the DataFrame
    df['prediction'] = predictions

    # Save the predictions to a new CSV file
    output_csv = input_csv.replace('.csv', '_predictions.csv')
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")



if __name__ == "__main__":
    # Path to the input CSV file
    input_csv = "data/preprocessed_features.csv"  # Replace with your CSV file path

    # Path to the pre-trained GAT model
    model_path = "gat_network_security_model.pth"  # Replace with your model path

    # Perform predictions
    predict_on_csv(input_csv, model_path)
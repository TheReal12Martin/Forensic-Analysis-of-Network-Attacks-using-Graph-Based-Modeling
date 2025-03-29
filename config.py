class Config:
    # Data Paths
    CSV_FILE = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
    FEATURES_CSV = None  # Not needed since we'll hardcode features
    
    # Label Configuration
    LABEL_COLUMN = 'Label'
    LABEL_MAPPING = {
        'Benign': 0,
        'DDoS': 1,
        'DDoS attacks-LOIC-HTTP': 1,
        'DDOS': 1
    }
    
    # CIC-IDS2018 Specific Features
    NUMERIC_FEATURES = [
        'Flow Duration', 
        'Tot Fwd Pkts',
        'Tot Bwd Pkts',
        'TotLen Fwd Pkts',
        'TotLen Bwd Pkts',
        'Flow Byts/s',
        'Flow Pkts/s',
        'Fwd Pkt Len Max',
        'Bwd Pkt Len Max',
        'Pkt Len Mean',
        'FIN Flag Cnt'
    ]
    
    CATEGORICAL_FEATURES = ['Protocol']
    
    # Label configuration
    LABEL_COLUMN = 'Label'

    # IP column names
    SRC_IP_COL = 'Src IP'
    DST_IP_COL = 'Dst IP'

    CLASS_NAMES = ['Benign', 'Malicious']

    # Add these to your existing config
    MAX_FEATURE_VALUE = 1e12  # For capping extreme values
    INF_REPLACEMENT = 1e12    # For replacing infinity
    
    # Graph Construction
    MIN_EDGE_WEIGHT = 1
    MAX_NODE_DEGREE = 1000
    
    # Model Architecture
    HIDDEN_CHANNELS = 128
    NUM_CLASSES = 2
    DROPOUT = 0.3
    HEADS = 8
    GAT_LAYERS = 2
    
    # Training
    TEST_RATIO = 0.2
    VAL_RATIO = 0.2
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    MIN_DELTA = 0.001
    PATIENCE = 15
    CLASS_WEIGHTS = [10.0, 1.0]  # Higher weight for DDoS
    RANDOM_STATE = 42

    # Memory optimization
    CHUNK_SIZE = 50000  # Rows per chunk
    # Memory Limits
    MAX_RAM_GB = 12  # Set to 80% of your available RAM
    NODE_CHUNK_SIZE = 5000  # Reduce if still crashing
    EDGE_CHUNK_SIZE = 100000
    
    # Downsampling (if needed)
    MAX_SAMPLES = 2000000  # Limit total samples
    DOWNSAMPLE_RATIO = 0.5  # Random sampling ratio
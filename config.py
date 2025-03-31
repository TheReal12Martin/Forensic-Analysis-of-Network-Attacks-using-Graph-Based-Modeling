class Config:
    # Updated Data Configuration
    DATA_FOLDER = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018"
    BENIGN_PATTERN = "*benign*.csv"  # Pattern to match benign files
    MALICIOUS_PATTERNS = ["*bot*.csv",
                         "*dos*.csv",
                         "*BF*.csv",
                         "*SQL*.csv",
                         "*infiltration*.csv",
                         "*DoS*.csv",
                         "*loic*.csv",
                         "*hoic*.csv"]
    
    # Label Configuration
    LABEL_MAPPING = {
        'benign': 0,
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 1,
        'Brute_Force_Web': 1,
        'Infiltration': 1,
        'DoS_HULK': 1,
        'DoS_SlowHTTP': 1,
        'Brute_Force_XSS': 1,
        'SQL_Injection': 1,
        'DoS_Golden_Eye': 1,
        'DoS_Slowloris': 1,
        'DDoS_Loic_HTTP': 1,
        'Brute_Force_FTP': 1,
        'Brute_Force_SSH': 1,
        'DDoS_HOIC': 1
    }
    
    # Feature Configuration (adjust based on your dataset)
    NUMERIC_FEATURES = [
        'duration',
        'packets_count',
        'total_payload_bytes',
        'payload_bytes_mean',
        'payload_bytes_std',
        'fwd_packets_count',
        'bwd_packets_count',
        'fwd_total_payload_bytes',
        'bwd_total_payload_bytes',
        'bytes_rate',
        'packets_rate',
        'fwd_packets_rate'
    ]
    
    CATEGORICAL_FEATURES = ['protocol']
    
    # Rest of your configuration remains the same...
    MAX_SAMPLES = 8_000_000  # Limit total samples
    MAX_PARTITION_SIZE_GB = 2.0
    CHUNK_SIZE = 100_000  # Rows per chun

    MAX_GRAPH_NODES = 50000  # Maximum nodes per graph
    MAX_GRAPH_EDGES = 200000  # Maximum edges per graph
    
    CLASS_NAMES = {'Benign': 0, 'Malicious': 1}

    CATEGORICAL_FEATURES = ['protocol']
    
    # IP columns
    SRC_IP_COL = 'src_ip'
    DST_IP_COL = 'dst_ip'
    
    # Calculated dimensions
    PROTOCOL_DIM = 3  # TCP, UDP, ICMP
    INPUT_FEATURES = len(NUMERIC_FEATURES) + PROTOCOL_DIM
    
    # Graph Construction
    MIN_EDGE_WEIGHT = 1
    MAX_NODE_DEGREE = 1000
    
    # Model Architecture
    HIDDEN_CHANNELS = 64
    NUM_CLASSES = 2
    DROPOUT = 0.5
    HEADS = 4
    GAT_LAYERS = 3
    
    # Training
    TEST_RATIO = 0.2
    VAL_RATIO = 0.2
    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MIN_DELTA = 0.005
    PATIENCE = 20
    CLASS_WEIGHTS = [1.0, 10.0]  # Higher weight for attacks
    RANDOM_STATE = 42
    BALANCE_RATIO = 0.5  # Target ratio of benign:malicious samples
    
    # Feature processing
    MAX_FEATURE_VALUE = 1e12
    INF_REPLACEMENT = 1e12
    
    # Focal Loss Parameters
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0

    @classmethod
    def update_feature_dimensions(cls, actual_features):
        """Update feature dimensions if automatic detection differs"""
        if actual_features != cls.INPUT_FEATURES:
            print(f"Updating INPUT_FEATURES from {cls.INPUT_FEATURES} to {actual_features}")
            cls.INPUT_FEATURES = actual_features
            cls.PROTOCOL_DIM = actual_features - len(cls.NUMERIC_FEATURES)
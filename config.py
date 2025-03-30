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
        'FIN Flag Cnt'  # 11 numeric features
    ]
    
    CATEGORICAL_FEATURES = ['Protocol']  # 1 categorical feature
    
    # Calculated feature dimensions (11 numeric + one-hot encoded Protocol)
    # Update PROTOCOL_DIM based on your actual Protocol categories (4 in your case)
    PROTOCOL_DIM = 4  # Number of unique Protocol values in your data
    INPUT_FEATURES = len(NUMERIC_FEATURES) + PROTOCOL_DIM  # 11 + 4 = 15
    
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

    # Focal Loss Parameters
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0
    
    # Data Balancing
    BALANCE_RATIO = 0.3  # Target ratio of benign:malicious samples


    @classmethod
    def update_feature_dimensions(cls, actual_features):
        """Update feature dimensions if automatic detection differs"""
        if actual_features != cls.INPUT_FEATURES:
            print(f"Updating INPUT_FEATURES from {cls.INPUT_FEATURES} to {actual_features}")
            cls.INPUT_FEATURES = actual_features
            cls.PROTOCOL_DIM = actual_features - len(cls.NUMERIC_FEATURES)
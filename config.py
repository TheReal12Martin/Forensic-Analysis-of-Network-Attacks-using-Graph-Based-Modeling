import os

import torch


class Config:
    # Updated Data Configuration
    DATA_FOLDER = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018"

    DATA_PATTERN = "*merged*.csv"
    
    # Label Configuration
    LABEL_MAPPING = {
        'benign': 0,
        'BENIGN': 0,
        'Benign': 0,
        'Bot': 1,
        'DDoS': 1,
        'Brute_Force_Web': 1,
        'Infiltration': 1,
        'Dos_Hulk': 1,
        'Dos_Slowhttp': 1,
        'Brute_Force_Xss': 1,
        'Sql_Injection': 1,
        'Dos_Golden_Eye': 1,
        'Dos_Slowloris': 1,
        'Ddos_Loic_Http': 1,
        'Brute_Force_Ftp': 1,
        'Brute_Force_Ssh': 1,
        'Ddos_Hoic': 1,
        0: 0,
        1: 1,
        '0': 0,
        '1': 1
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
    
    MAX_PARTITION_SIZE_GB = 1.0  # Process up to 1GB at a time
    MAX_SAMPLES = 1_000_000  # Limit total samples
    CHUNK_SIZE = 100_000  # Rows per chun

    MAX_GRAPH_NODES = 5000  # Maximum nodes per graph
    MAX_NODE_DEGREE = 5000
    MAX_GRAPH_EDGES = 5000  # Maximum edges per graph
    MIN_EDGE_WEIGHT = 1
    BATCH_SIZE = 256
    GRADIENT_ACCUMULATION_STEPS = 4

    MIN_TEST_SAMPLES = 100       # Minimum samples per class in test set
    MIN_VAL_ACC = 0.7

    MIN_SAMPLES_PER_CLASS = 1000  # Minimum samples per class to process
    K_NEIGHBORS = 10               # For KNN graph construction
    SIMILARITY_THRESHOLD = 0.7    # Cosine similarity threshold
    
    CLASS_NAMES = {'Benign': 0, 'Malicious': 1}

    CATEGORICAL_FEATURES = ['protocol']
    
    # IP columns
    SRC_IP_COL = 'src_ip'
    DST_IP_COL = 'dst_ip'
    
    # Calculated dimensions
    PROTOCOL_DIM = 3  # TCP, UDP, ICMP
    INPUT_FEATURES = len(NUMERIC_FEATURES) + PROTOCOL_DIM
    
    
    # Model Architecture
    HIDDEN_CHANNELS = 256
    NUM_CLASSES = 2
    DROPOUT = 0.7
    HEADS = 8
    GAT_LAYERS = 3
    
    # Training
    TEST_RATIO = 0.2
    VAL_RATIO = 0.2
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MIN_DELTA = 0.005
    PATIENCE = 20
    CLASS_WEIGHTS = [1.0, 5.0]  # Higher weight for attacks
    RANDOM_STATE = 42
    BALANCE_RATIO = 0.5  # Target ratio of benign:malicious samples
    
    # Feature processing
    MAX_FEATURE_VALUE = 1e12
    INF_REPLACEMENT = 1e12
    
    # Focal Loss Parameters
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0

    CPU_THREADS = 2  # Optimal for most systems
    FORCE_SINGLE_CORE = False  # Set True if still seeing high CPU

    @classmethod
    def update_feature_dimensions(cls, actual_features):
        """Update feature dimensions if automatic detection differs"""
        if actual_features != cls.INPUT_FEATURES:
            print(f"Updating INPUT_FEATURES from {cls.INPUT_FEATURES} to {actual_features}")
            cls.INPUT_FEATURES = actual_features
            cls.PROTOCOL_DIM = actual_features - len(cls.NUMERIC_FEATURES)
    @classmethod
    def apply_cpu_limits(cls):
        """Apply CPU thread limits"""
        if cls.FORCE_SINGLE_CORE:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
        else:
            torch.set_num_threads(cls.CPU_THREADS)
class Config:
    # Data Paths
    CSV_FILE = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/train_UNSW_NB15.csv"
    FEATURES_CSV = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/NUSW-NB15_features.csv"

    # Data Processing
    SAMPLE_SIZE = 10000  # Reduced from 30000
    MAX_FEATURES = 20  # Reduced from 50
    RANDOM_STATE = 42

    # Core Features Only
    NUMERIC_FEATURES = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'Sload', 'Dload']
    CATEGORICAL_FEATURES = ['proto', 'service']

    # Graph Parameters
    MIN_EDGE_WEIGHT = 1
    MAX_NODE_DEGREE = 50  # Reduced from 1000

    # Simplified Model
    HIDDEN_CHANNELS = 64
    NUM_CLASSES = 2
    DROPOUT = 0.5
    HEADS = 2
    GAT_LAYERS = 1
    WEIGHT_DECAY = 5e-4
    MIN_DELTA = 0.005

    # Training
    EPOCHS = 30
    LEARNING_RATE = 0.001
    TEST_RATIO = 0.2
    VAL_RATIO = 0.1
    PATIENCE = 10
    CLASS_WEIGHTS = [1.0, 10.0]  # Moderate class weighting

    # Label Configuration (NEW)
    LABEL_COLUMN = 'Label'  # Column name containing labels in your data
    LABEL_MAPPING = {       # How to interpret different label values
        'benign': 0,
        'normal': 0,
        '0': 0,
        'malicious': 1,
        'attack': 1,
        '1': 1
    }

    BALANCE_CLASSES = True  # Set to False to disable class balancing
    MAX_SAMPLES_PER_CLASS = 30000  # Maximum samples per class
    MIN_SAMPLES_PER_CLASS = 10000  # Minimum samples per class (for minority class)
    MAX_FEATURES = 50
    RANDOM_STATE = 42

    # Feature Configuration
    NUMERIC_FEATURES = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
                       'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                       'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                       'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat']
    CATEGORICAL_FEATURES = ['proto', 'service', 'state']
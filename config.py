class Config:
    # Data Paths
    CSV_FILE = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/train_UNSW_NB15.csv"
    FEATURES_CSV = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/NUSW-NB15_features.csv"

    # Data Processing
    IP_ANONYMIZE = True
    IP_HASH_LENGTH = 8
    MIN_SAMPLES_PER_CLASS = 5000
    RANDOM_STATE = 42

    # Model Architecture (Reduced Capacity)
    HIDDEN_CHANNELS = 64  # Reduced from 64
    NUM_CLASSES = 2
    DROPOUT = 0.2  # Reduced from 0.4
    HEADS = 2  # Number of attention heads

    # Training Parameters (Conservative)
    EPOCHS = 200
    LEARNING_RATE = 0.0005  # Reduced from 0.001
    WEIGHT_DECAY = 1e-3
    TEST_RATIO = 0.2
    VAL_RATIO = 0.1
    PATIENCE = 5  # For early stopping
    MIN_DELTA = 0.001  # Minimum improvement threshold

    # Neighbor Sampling
    BATCH_SIZE = 4096  # Increased for stability
    NUM_NEIGHBORS = [15, 8]  # Smaller neighborhood
    NUM_WORKERS = 4  # Data loading threads

    # Loss Weights (Adjust based on your class distribution)
    CLASS_WEIGHTS = [1.0, 3.0]  # [weight_for_class_0, weight_for_class_1]
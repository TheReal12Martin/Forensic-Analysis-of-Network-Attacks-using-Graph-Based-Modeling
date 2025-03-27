class Config:
    #Data Path
    CSV_FILE = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/train_UNSW_NB15.csv"
    FEATURES_CSV = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/NUSW-NB15_features.csv"

    #Model Hyperparameters
    HIDDEN_CHANNELS = 16
    HEADS = 8
    NUM_CLASSES = 2
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200

    #Training Settings
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    VALIDATION = {
        'min_leakage_score': 0.05,  # Fail if IP leakage >5%
        'temporal_test_ratio': 0.2,
        'feature_importance_trials': 3,
        'noise_levels': [0.01, 0.05, 0.1]
    }

    MIN_SAMPLES_PER_CLASS = 5  # Minimum samples per class after balancing
    MIN_NODES_FOR_SPLIT = 10   # Minimum nodes required for train/val/test split
    RANDOM_STATE = 42

    # Security
    IP_ANONYMIZE = True
    MIN_IP_OCCURRENCES = 3  # Minimum times IP must appear
    IP_HASH_LENGTH = 8      # Length of anonymized IP hashes
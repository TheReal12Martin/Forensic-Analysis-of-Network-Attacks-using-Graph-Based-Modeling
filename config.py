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

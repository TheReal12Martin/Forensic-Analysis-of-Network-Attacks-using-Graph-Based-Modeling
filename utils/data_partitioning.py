import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from config import Config
import gc

def partition_and_process(input_csv, partitions=4, output_dir="data_partitions"):
    """Split large CSV into processed partitions with validation"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # First verify the CSV has required columns
        sample = pd.read_csv(input_csv, nrows=1)
        missing_cols = [col for col in Config.NUMERIC_FEATURES + Config.CATEGORICAL_FEATURES 
                      if col not in sample.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        del sample
        gc.collect()

        # Get total rows safely
        with open(input_csv) as f:
            total_rows = sum(1 for _ in f) - 1  # exclude header
        
        if total_rows <= 0:
            raise ValueError("CSV file is empty or has only header")
        
        rows_per_part = total_rows // partitions
        partition_files = []
        
        # Read and process chunks
        for i in range(partitions):
            part_file = os.path.join(output_dir, f"partition_{i}.npz")
            print(f"Creating partition {i+1}/{partitions}")
            
            # Read chunk with column verification
            try:
                chunk = pd.read_csv(
                    input_csv,
                    skiprows=i * rows_per_part + 1,
                    nrows=rows_per_part,
                    names=pd.read_csv(input_csv, nrows=0).columns,  # preserve header
                    dtype={
                        **{col: np.float32 for col in Config.NUMERIC_FEATURES},
                        **{col: 'category' for col in Config.CATEGORICAL_FEATURES},
                        Config.LABEL_COLUMN: 'string'
                    }
                )
            except pd.errors.EmptyDataError:
                raise ValueError(f"Partition {i} is empty - check your partitioning")

            # Verify we got data
            if len(chunk) == 0:
                raise ValueError(f"Partition {i} is empty - check your partitioning")
            
            # Process numeric features
            numeric = chunk[Config.NUMERIC_FEATURES].replace([np.inf, -np.inf], 1e12).fillna(0)
            numeric = np.clip(numeric, -1e12, 1e12)
            numeric = StandardScaler().fit_transform(numeric)
            
            # Process categorical features
            categorical = pd.get_dummies(chunk[Config.CATEGORICAL_FEATURES], dummy_na=True)
            
            # Process labels
            labels = (chunk[Config.LABEL_COLUMN]
                     .str.strip()
                     .str.upper()
                     .map(Config.LABEL_MAPPING)
                     .fillna(1)
                     .astype(np.int8))
            
            # Save processed partition
            np.savez_compressed(
                part_file,
                features=np.hstack([numeric, categorical]),
                labels=labels.values,
                src_ips=chunk[Config.SRC_IP_COL].values if Config.SRC_IP_COL in chunk.columns else None,
                dst_ips=chunk[Config.DST_IP_COL].values if Config.DST_IP_COL in chunk.columns else None
            )
            
            partition_files.append(part_file)
            del chunk, numeric, categorical, labels
            gc.collect()
        
        return partition_files
        
    except Exception as e:
        # Cleanup partial results
        if 'partition_files' in locals():
            for f in partition_files:
                if os.path.exists(f):
                    os.remove(f)
        raise ValueError(f"Partitioning failed: {str(e)}")
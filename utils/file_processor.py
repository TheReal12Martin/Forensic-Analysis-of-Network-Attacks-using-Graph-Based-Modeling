import os
import glob
import pandas as pd
import numpy as np
from config import Config
import gc

def get_dataset_files():
    """Get all files matching the data pattern"""
    return glob.glob(os.path.join(Config.DATA_FOLDER, Config.DATA_PATTERN))

def split_large_file(file_path, max_size_gb=1.0):
    """Split large CSV into chunks of approximately max_size_gb"""
    # Estimate rows per chunk
    file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
    if file_size <= max_size_gb:
        return [file_path]
    
    # Get approximate rows per chunk
    with pd.read_csv(file_path, chunksize=1000) as reader:
        sample_chunk = next(reader)
    bytes_per_row = sample_chunk.memory_usage(deep=True).sum() / 1000
    rows_per_chunk = int((max_size_gb * 1024**3) / bytes_per_row)
    
    # Create chunks
    chunks = []
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=rows_per_chunk)):
        chunk_path = f"{file_path}_part{i}.csv"
        chunk.to_csv(chunk_path, index=False)
        chunks.append(chunk_path)
        del chunk
        gc.collect()
    
    return chunks

def prepare_all_files():
    """Prepare all files by splitting large ones"""
    all_files = []
    for file in get_dataset_files():
        all_files.extend(split_large_file(file, Config.MAX_PARTITION_SIZE_GB))
    return all_files
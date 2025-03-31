import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config
import gc
from tqdm import tqdm
import warnings

def get_memory_usage():
    """Returns memory usage in MB (more accurate)"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def load_chunk_safely(file_path):
    """Load CSV chunks with strict dtype handling"""
    # Identify problematic columns to exclude or handle specially
    problematic_cols = {
        '213': 'float32',  # Force specific types for known problematic columns
        '214': 'float32',
        'flow_id': 'str',
        'timestamp': 'str'
    }
    
    # Base dtype specification
    dtype = {
        'src_ip': 'category',
        'dst_ip': 'category',
        'protocol': 'category',
        **{col: 'float32' for col in Config.NUMERIC_FEATURES},
        **problematic_cols
    }
    
    # Columns to exclude completely
    exclude_cols = [str(i) for i in range(200,250) if str(i) not in problematic_cols]
    
    try:
        return pd.read_csv(
            file_path,
            chunksize=Config.CHUNK_SIZE,
            dtype=dtype,
            usecols=lambda x: x not in exclude_cols,
            low_memory=False,
            engine='c'
        )
    except Exception as e:
        warnings.warn(f"Error loading {file_path}: {str(e)}")
        return []

def count_rows_safely(files):
    """Count rows without loading full data"""
    counts = []
    for f in files:
        try:
            with open(f) as file:
                counts.append(sum(1 for _ in file) - 1)  # Subtract header
        except Exception as e:
            warnings.warn(f"Error counting {f}: {str(e)}")
            counts.append(0)
    return sum(counts)

def process_chunk(chunk):
    """Clean and prepare a data chunk"""
    # Handle protocol
    if 'protocol' in chunk.columns:
        chunk['protocol'] = chunk['protocol'].astype(str).str.upper().str.strip().fillna('UNKNOWN')
    
    # Clean numeric columns
    for col in Config.NUMERIC_FEATURES:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0).astype('float32')
    
    return chunk

def balanced_data_generator():
    """Memory-optimized generator with better error handling"""
    from glob import glob
    
    # Get files with validation
    benign_files = glob(os.path.join(Config.DATA_FOLDER, Config.BENIGN_PATTERN))
    malicious_files = []
    for pattern in Config.MALICIOUS_PATTERNS:
        malicious_files.extend(glob(os.path.join(Config.DATA_FOLDER, pattern)))
    
    # Verify we have both classes
    if not benign_files or not malicious_files:
        raise ValueError("Need both benign and malicious files for balanced dataset")

    # Process files in an interleaved manner to maintain balance
    while True:
        # Yield benign chunks
        for file_path in benign_files:
            for chunk in load_chunk_safely(file_path):
                chunk = process_chunk(chunk)
                chunk['label'] = 0  # Benign
                yield chunk
                gc.collect()
        
        # Yield malicious chunks (with sampling if needed)
        for file_path in malicious_files:
            for chunk in load_chunk_safely(file_path):
                chunk = process_chunk(chunk)
                chunk = chunk.sample(frac=Config.BALANCE_RATIO, random_state=Config.RANDOM_STATE)
                chunk['label'] = 1  # Malicious
                yield chunk
                gc.collect()

def process_and_save_partitions(data_gen, output_dir="data_partitions"):
    os.makedirs(output_dir, exist_ok=True)
    scaler = StandardScaler()
    protocol_dummies = None
    partition_data = []
    partition_num = 0
    total_samples = 0  # Track total samples to enforce MAX_SAMPLES

    for chunk in tqdm(data_gen, desc="Processing Partitions"):
        # Enforce MAX_SAMPLES limit
        if total_samples >= Config.MAX_SAMPLES:
            print(f"Stopping early: Reached MAX_SAMPLES ({Config.MAX_SAMPLES})")
            break

        # Process chunk (existing code)
        numeric = scaler.partial_fit(chunk[Config.NUMERIC_FEATURES]).transform(chunk[Config.NUMERIC_FEATURES])
        protocol = pd.get_dummies(chunk['protocol'].fillna('UNKNOWN'), prefix='proto', dtype=np.int8)
        if protocol_dummies is None:
            protocol_dummies = protocol.columns
        protocol = protocol.reindex(columns=protocol_dummies, fill_value=0)
        features = np.hstack([numeric, protocol.values]).astype(np.float32)

        partition_data.append({
            'features': features,
            'labels': chunk['label'].values.astype(np.int8),
            'src_ips': chunk['src_ip'].astype('U15').values,
            'dst_ips': chunk['dst_ip'].astype('U15').values
        })
        total_samples += len(chunk)  # Update counter

        # Save partition if memory limit reached OR too many samples
        if (get_memory_usage() > Config.MAX_PARTITION_SIZE_GB * 900 or
            len(partition_data) > 10):  # Force save after 10 chunks
            save_partition(partition_data, output_dir, partition_num)
            partition_num += 1
            partition_data = []
            gc.collect()

    # Save final partition if remaining
    if partition_data:
        save_partition(partition_data, output_dir, partition_num)

    return sorted(glob.glob(os.path.join(output_dir, "partition_*.npz")))

def save_partition(partition_data, output_dir, partition_num):
    """Save partition with validation"""
    try:
        # Combine data
        features = np.vstack([d['features'] for d in partition_data])
        labels = np.concatenate([d['labels'] for d in partition_data])
        src_ips = np.concatenate([d['src_ips'] for d in partition_data])
        dst_ips = np.concatenate([d['dst_ips'] for d in partition_data])

        # Validate
        assert len(features) == len(labels) == len(src_ips) == len(dst_ips), "Length mismatch"

        # Save
        path = os.path.join(output_dir, f"partition_{partition_num}.npz")
        np.savez_compressed(
            path,
            features=features.astype(np.float32),
            labels=labels.astype(np.int8),
            src_ips=src_ips.astype('U15'),
            dst_ips=dst_ips.astype('U15')
        )
        print(f"Saved partition {partition_num} with {len(features):,} samples")

    except Exception as e:
        print(f"Error saving partition: {str(e)}")
        raise
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
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

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

def process_chunk(chunk_df):
    """Process chunks with proper feature handling"""
    try:
        # Convert to dict if DataFrame
        if hasattr(chunk_df, 'to_dict'):
            chunk = {k: v.values if hasattr(v, 'values') else v 
                    for k, v in chunk_df.to_dict('series').items()}
        else:
            chunk = chunk_df

        # Initialize output
        processed = {
            'features': None,
            'labels': np.array([], dtype=np.int8),
            'src_ips': np.array([], dtype='U15'),
            'dst_ips': np.array([], dtype='U15'),
            'protocol': np.array([], dtype='U15')
        }

        # Skip empty chunks
        if not chunk.get('src_ip', []):
            return None

        # Process labels
        if 'label' not in chunk:
            raise ValueError("Label column missing")
        processed['labels'] = np.array([
            Config.LABEL_MAPPING.get(str(x).strip().title(), 1) 
            for x in chunk['label']
        ], dtype=np.int8)

        # Process IPs
        processed['src_ips'] = np.array(chunk['src_ip'], dtype='U15')
        processed['dst_ips'] = np.array(chunk['dst_ip'], dtype='U15')
        
        # Process protocol
        if 'protocol' in chunk:
            processed['protocol'] = np.array(chunk['protocol'], dtype='U15')
        
        # Process numeric features
        numeric_feats = []
        for feat in Config.NUMERIC_FEATURES:
            if feat in chunk:
                col_data = np.array(chunk[feat], dtype=np.float32)
                col_data = np.nan_to_num(col_data)
                numeric_feats.append(col_data)
        
        if numeric_feats:
            processed['features'] = np.column_stack(numeric_feats)

        return processed

    except Exception as e:
        print(f"⚠️ Chunk processing failed: {str(e)}")
        return None
    



def balanced_data_generator():
    """Generator that yields processed chunks"""
    all_files = glob.glob(os.path.join(Config.DATA_FOLDER, Config.DATA_PATTERN))
    np.random.shuffle(all_files)

    for file_path in all_files:
        try:
            for chunk_df in load_chunk_safely(file_path):
                processed = process_chunk(chunk_df)
                if processed is None:
                    continue
                    
                # Verify we have data
                if len(processed['labels']) == 0:
                    continue
                    
                yield processed
                    
        except Exception as e:
            print(f"⚠️ Error processing {os.path.basename(file_path)}: {str(e)}")
            continue



def save_partition(partition_data, output_dir, partition_num):
    """Save partition only if it contains both classes"""
    try:
        # Combine data
        features = np.vstack([d['features'] for d in partition_data])
        labels = np.concatenate([d['labels'] for d in partition_data])
        src_ips = np.concatenate([d['src_ips'] for d in partition_data])
        dst_ips = np.concatenate([d['dst_ips'] for d in partition_data])

        # Validate we have both classes
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            print(f"⚠️ Discarding partition {partition_num} (only class {unique_labels[0]} with {counts[0]} samples)")
            return None

        # Validate array lengths
        if not (len(features) == len(labels) == len(src_ips) == len(dst_ips)):
            raise ValueError("Length mismatch between features and labels")

        # Save valid partition
        path = os.path.join(output_dir, f"partition_{partition_num}.npz")
        np.savez_compressed(
            path,
            features=features.astype(np.float32),
            labels=labels.astype(np.int8),
            src_ips=src_ips.astype('U15'),
            dst_ips=dst_ips.astype('U15')
        )
        print(f"✅ Saved partition {partition_num} (Class 0: {counts[0]}, Class 1: {counts[1]})")
        return path

    except Exception as e:
        print(f"❌ Failed to save partition {partition_num}: {str(e)}")
        return None

def process_and_save_partitions(data_gen, output_dir="data_partitions"):
    """Process data chunks into validated partitions"""
    os.makedirs(output_dir, exist_ok=True)
    scaler = StandardScaler()
    protocol_dummies = None
    partition_data = []
    partition_num = 0
    valid_partitions = []
    
    for chunk in tqdm(data_gen, desc="Creating Partitions"):
        try:
            # Skip empty chunks
            if not chunk or len(chunk['labels']) == 0:
                continue

            # Process numeric features
            numeric = scaler.partial_fit(chunk['features']).transform(chunk['features'])
            
            # Process protocol
            protocol = pd.get_dummies(
                pd.Series(chunk.get('protocol', ['UNKNOWN']*len(chunk['labels']))),
                prefix='proto'
            )
            if protocol_dummies is None:
                protocol_dummies = protocol.columns
            protocol = protocol.reindex(columns=protocol_dummies, fill_value=0)
            
            features = np.hstack([numeric, protocol.values]).astype(np.float32)
            
            partition_data.append({
                'features': features,
                'labels': chunk['labels'],
                'src_ips': chunk['src_ips'],
                'dst_ips': chunk['dst_ips']
            })

            # Save when reaching size limit
            if (get_memory_usage() > Config.MAX_PARTITION_SIZE_GB * 900 or
                len(partition_data) >= 10):
                
                saved_path = save_partition(partition_data, output_dir, partition_num)
                if saved_path:
                    valid_partitions.append(saved_path)
                    partition_num += 1
                partition_data = []
                gc.collect()
                
        except Exception as e:
            print(f"⚠️ Chunk processing error: {str(e)}")
            continue

    # Save final partition if valid
    if partition_data:
        saved_path = save_partition(partition_data, output_dir, partition_num)
        if saved_path:
            valid_partitions.append(saved_path)

    print(f"\nCreated {len(valid_partitions)} valid partitions out of {partition_num + 1} attempts")
    return sorted(valid_partitions)
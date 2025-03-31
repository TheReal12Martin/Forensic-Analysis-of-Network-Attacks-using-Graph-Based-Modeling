import os
import gc
from config import Config
from utils.data_loader import process_and_save_partitions, balanced_data_generator
import pandas as pd

def validate_input_files():
    """Validate input files using multiple patterns"""
    from glob import glob
    import os
    
    # Get benign files
    benign_files = glob(os.path.join(Config.DATA_FOLDER, Config.BENIGN_PATTERN))
    
    # Get malicious files using all patterns
    malicious_files = []
    for pattern in Config.MALICIOUS_PATTERNS:
        matched_files = glob(os.path.join(Config.DATA_FOLDER, pattern))
        # Exclude benign files and already matched files
        matched_files = [f for f in matched_files 
                       if "benign" not in os.path.basename(f).lower() and
                       f not in malicious_files]
        malicious_files.extend(matched_files)
    
    if not benign_files:
        available = [f for f in os.listdir(Config.DATA_FOLDER) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"No benign files found matching {Config.BENIGN_PATTERN}\n"
            f"Available CSV files: {available}"
        )
    
    if not malicious_files:
        available = [f for f in os.listdir(Config.DATA_FOLDER) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"No malicious files found using patterns: {Config.MALICIOUS_PATTERNS}\n"
            f"Available CSV files: {available}"
        )
    
    print(f"Found {len(benign_files)} benign files")
    print(f"Found {len(malicious_files)} malicious files:")
    for f in malicious_files[:10]:  # Show first 10 for verification
        print(f"- {os.path.basename(f)}")
    
    return True

def clean_partition_directory(output_dir):
    """Ensure clean output directory"""
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("partition_") and f.endswith(".npz"):
                os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

def partition_and_process(output_dir="data_partitions"):
    """
    Main function to process dataset into balanced partitions
    Returns:
        List of paths to created partition files
    """
    partition_files = []
    
    try:
        # 1. Validation
        print("Validating input files...")
        validate_input_files()
        
        # 2. Prepare output directory
        clean_partition_directory(output_dir)
        
        # 3. Create balanced data generator
        print("Creating balanced dataset generator...")
        data_gen = balanced_data_generator()
        
        # 4. Process and save partitions
        print(f"Creating partitions (max {Config.MAX_PARTITION_SIZE_GB}GB each)...")
        partition_files = process_and_save_partitions(
            data_gen,
            output_dir=output_dir,
            #max_size_mb=Config.MAX_PARTITION_SIZE_GB * 900  # 90% of target for safety
        )
        
        # 5. Verify output
        if not partition_files:
            raise RuntimeError("No partitions were created - check data and memory limits")
        
        created_sizes = [os.path.getsize(f)/(1024**3) for f in partition_files]
        print(f"Successfully created {len(partition_files)} partitions. Sizes (GB): {created_sizes}")
        
        return partition_files
        
    except Exception as e:
        print(f"\nError during partitioning: {str(e)}")
        
        # Clean up any partial results
        if partition_files:
            print("Cleaning up partial partitions...")
            for f in partition_files:
                if os.path.exists(f):
                    os.remove(f)
        
        raise
    finally:
        gc.collect()

def get_partition_files(output_dir="data_partitions"):
    """Get sorted list of partition files with validation"""
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    partition_files = sorted([
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.startswith("partition_") and f.endswith(".npz")
    ])
    
    if not partition_files:
        raise FileNotFoundError(f"No partition files found in {output_dir}")
    
    return partition_files
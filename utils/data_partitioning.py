import os
import gc
from config import Config
from utils.data_loader import process_and_save_partitions, balanced_data_generator
import pandas as pd

def validate_input_files():
    """Validate that input files exist"""
    from glob import glob
    import os
    
    # Get all data files (handle both string and list patterns)
    if isinstance(Config.DATA_PATTERN, list):
        data_files = []
        for pattern in Config.DATA_PATTERN:
            data_files.extend(glob(os.path.join(Config.DATA_FOLDER, pattern)))
    else:
        data_files = glob(os.path.join(Config.DATA_FOLDER, Config.DATA_PATTERN))
    
    if not data_files:
        available = [f for f in os.listdir(Config.DATA_FOLDER) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"No data files found matching {Config.DATA_PATTERN}\n"
            f"Available files: {available}"
        )
    
    print(f"Found {len(data_files)} data files")
    for f in data_files[:5]:  # Show first 5 for verification
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
    """Main partitioning function for mixed traffic"""
    try:
        print("Validating input files...")
        validate_input_files()
        
        print("Creating balanced dataset generator...")
        data_gen = balanced_data_generator()
        
        print("Creating partitions...")
        partition_files = process_and_save_partitions(
            data_gen,
            output_dir=output_dir
        )
        
        if not partition_files:
            raise RuntimeError("No partitions were created")
        
        print(f"Successfully created {len(partition_files)} partitions")
        return partition_files
        
    except Exception as e:
        print(f"Error during partitioning: {str(e)}")
        raise

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
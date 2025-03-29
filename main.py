import sys
import os
from train import train_partition
from evaluate import evaluate_partition
from utils.data_partitioning import partition_and_process
from config import Config
import gc

def main():
    try:
        print("=== Starting Partitioned Training Pipeline ===")
        
        # Step 1: Partition and process data
        print("\n=== Partitioning and Preprocessing Data ===")
        partition_files = partition_and_process(
            Config.CSV_FILE,
            partitions=4,  # Split into 4 chunks
            output_dir="data_partitions"
        )
        
        # Step 2: Process each partition
        models = []
        for i, partition_file in enumerate(partition_files):
            print(f"\n=== Processing Partition {i+1}/{len(partition_files)} ===")
            model, data = train_partition(partition_file)
            models.append((model, data))
            
            # Evaluate this partition immediately to free memory
            evaluate_partition(model, data)
            
            # Clean up
            del model, data
            gc.collect()
        
        # Optional: Combine models or analyze cross-partition results
        print("\n=== All partitions processed successfully ===")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
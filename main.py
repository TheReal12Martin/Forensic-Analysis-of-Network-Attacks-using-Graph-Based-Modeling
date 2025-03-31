import sys

import torch
from train import train_partition
from evaluate import evaluate_partition
from utils.data_partitioning import partition_and_process
from config import Config
import gc

def main():
    try:
        print("=== Starting Training Pipeline ===")
        
        # Step 1: Create balanced partitions
        print("\n=== Creating Partitions ===")
        partition_files = partition_and_process()
        
        # Step 2: Train on each partition
        for i, partition_file in enumerate(partition_files):
            print(f"\n=== Processing Partition {i+1}/{len(partition_files)} ===")
            try:
                model, data = train_partition(partition_file)
                evaluate_partition(model, data)
            except Exception as e:
                print(f"Failed on {partition_file}: {str(e)}")
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import torch
from config import Config
from train import train_partition
from evaluate import evaluate_model
from utils.data_partitioning import partition_and_process
import gc
import os

def main():
    print("=== Starting Training Pipeline ===")
    
    # 1. Create Partitions
    print("\n=== Creating Partitions ===")
    partition_files = partition_and_process()
    if not partition_files:
        raise RuntimeError("No valid partitions created!")

    # 2. Train & Evaluate on Each Partition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Training on {device} ===")

    for i, partition_file in enumerate(partition_files):
        print(f"\n=== Processing Partition {i+1}/{len(partition_files)} ===")
        print(f"File: {os.path.basename(partition_file)}")

        # Train
        model, data = train_partition(partition_file)
        
        # Evaluate only if training succeeded
        if model is not None and data is not None:
            evaluate_model(model, data, device)
        else:
            print(f"Skipping evaluation (training failed for {partition_file})")

        # Cleanup
        del model, data
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
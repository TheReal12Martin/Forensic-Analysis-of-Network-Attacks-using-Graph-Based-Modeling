import torch
from config import Config
from train import train_partition
from evaluate import evaluate_model
from utils.monitoring import initialize_monitoring, log_resource_usage
import gc
import os

from utils.data_partitioning import get_partition_files

def main():
    # Initialize monitoring and resource limits
    initialize_monitoring()
    Config.apply_cpu_limits()  # Fixed: Proper classmethod call
    
    print(f"=== Starting Training (CPU Threads: {torch.get_num_threads()}) ===")
    log_resource_usage("Start of training")
    
    # Get partitions
    partition_files = get_partition_files()
    if not partition_files:
        raise RuntimeError("No valid partitions found")
    
    device = torch.device('cpu')
    
    for i, partition_file in enumerate(partition_files):
        print(f"\n=== Processing Partition {i+1}/{len(partition_files)} ===")
        log_resource_usage(f"Before partition {i+1}")
        
        # Train
        model, data = train_partition(partition_file)
        
        # Evaluate
        if model is not None and data is not None:
            evaluate_model(model, data, device)
        
        # Cleanup
        del model, data
        gc.collect()
        log_resource_usage(f"After partition {i+1}")
    
    print("\n=== Training Completed ===")
    log_resource_usage("End of training")

if __name__ == "__main__":
    # Set lowest priority
    os.nice(19)
    main()
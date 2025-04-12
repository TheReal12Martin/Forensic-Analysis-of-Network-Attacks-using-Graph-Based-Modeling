import torch
from config import Config
from utils.pyg_conversion import convert_to_pyg_memory_safe
from utils.graph_construction import build_graph_from_partition
from train import train_model
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
    
    for i, pfile in enumerate(partition_files):
        print(f"\n=== Processing Partition {i+1}/{len(partition_files)} ===")
        log_resource_usage(f"Before partition {i+1}")
        
        # Build and convert graph
        graph, _ = build_graph_from_partition(pfile)
        data = convert_to_pyg_memory_safe(graph, device)
        
        if data:
            # Train and evaluate
            model, avg_acc = train_model(data, device)
            if model:
                evaluate_model(model, data, device)
                print(f"Average validation accuracy: {avg_acc:.2f}")
            
            # Cleanup
            del model, data, graph
            gc.collect()
        
        log_resource_usage(f"After partition {i+1}")
    
    print("\n=== Training Completed ===")
    log_resource_usage("End")

if __name__ == "__main__":
    os.nice(19)  # Lower priority
    main()
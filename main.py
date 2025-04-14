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
        
        try:
            # Graph building with size validation
            graph, _ = build_graph_from_partition(pfile)
            if graph is None:
                print(f"Skipping partition {i+1} - graph construction failed")
                log_resource_usage(f"Skipped partition {i+1}")
                continue
                
            if graph.number_of_nodes() < Config.MIN_GRAPH_NODES:
                print(f"Skipping partition {i+1} - only {graph.number_of_nodes()} nodes")
                continue
                
            # Conversion with validation
            data = convert_to_pyg_memory_safe(graph, device)
            if not data:
                print(f"Skipping partition {i+1} - conversion failed")
                continue
                
            # Training validation
            if sum(data.val_mask) < Config.MIN_VAL_SAMPLES:
                print(f"Skipping training - only {sum(data.val_mask)} validation samples")
                continue
                
            # Proceed with training
            model, avg_acc = train_model(data, device)
            if model:
                if sum(data.test_mask) >= Config.MIN_TEST_SAMPLES:
                    evaluate_model(model, data, device)
                else:
                    print(f"Not enough test samples ({sum(data.test_mask)}) for evaluation")
                    
                print(f"Average validation accuracy: {avg_acc:.2f}")
            
            # Cleanup
            del model, data, graph
            gc.collect()
            
        except Exception as e:
            print(f"Partition {i+1} failed: {str(e)}")
            continue
            
        log_resource_usage(f"After partition {i+1}")
    
    print("\n=== Training Completed ===")
    log_resource_usage("End")

if __name__ == "__main__":
    os.nice(19)  # Lower priority
    main()
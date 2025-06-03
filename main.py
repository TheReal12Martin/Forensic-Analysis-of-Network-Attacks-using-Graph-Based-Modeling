import numpy as np
import torch
from config import Config
from metrics_visualization import plot_combined_metrics_trend
from utils.pyg_conversion import convert_to_pyg_memory_safe
from utils.graph_construction import build_graph_from_partition
from train import train_model
from utils.monitoring import initialize_monitoring, log_resource_usage
from utils.diagnostics import verify_data_consistency, run_baseline_comparison
import gc
import os
from utils.data_partitioning import get_partition_files

def process_partition(pfile, device, partition_idx, total_partitions):
    """Enhanced partition processing with repair mechanisms"""
    print(f"\n=== Processing Partition {partition_idx}/{total_partitions} ===")
    log_resource_usage(f"Before partition {partition_idx}")

    try:
        # Initial processing attempt
        graph, _ = build_graph_from_partition(pfile)
        if graph is None:
            print(f"Skipping partition {partition_idx} - graph construction failed")
            return None

        data = convert_to_pyg_memory_safe(graph, device)
        if not data:
            print(f"Skipping partition {partition_idx} - conversion failed")
            return None

        # First verification pass
        data, is_valid = verify_data_consistency(data)
        
        # Repair attempt if needed
        if not is_valid:
            print(f"⚠️ Applying repair procedures to partition {partition_idx}")
            
            # Attempt 1: Rebuild with stricter parameters
            original_min_sim = Config.MIN_SIMILARITY
            original_max_neigh = Config.MAX_NEIGHBORS
            
            Config.MIN_SIMILARITY = min(0.9, Config.MIN_SIMILARITY * 1.2)
            Config.MAX_NEIGHBORS = max(3, Config.MAX_NEIGHBORS - 2)
            
            graph, _ = build_graph_from_partition(pfile)
            data = convert_to_pyg_memory_safe(graph, device)
            
            # Restore original config
            Config.MIN_SIMILARITY = original_min_sim
            Config.MAX_NEIGHBORS = original_max_neigh
            
            if not data:
                print(f"❌ Could not repair partition {partition_idx} - skipping")
                return None

        # Run baseline on first valid partition
        if partition_idx == 1:
            run_baseline_comparison(data)

        # Training and evaluation
        model, avg_acc, metrics = train_model(data, device, partition_idx)
        log_resource_usage(f"After partition {partition_idx}")
        
        return metrics

    except Exception as e:
        print(f"❌ Partition {partition_idx} failed: {str(e)}")
        return None
    finally:
        gc.collect()

def _final_aggregate_report(all_metrics):
    """Generate final aggregated reports"""
    print("\n=== Training Completed ===")
    
    # Aggregate validation metrics
    val_metrics = [m['val'] for m in all_metrics if m and m['val']]
    if val_metrics:
        print("\n🔍 Aggregate Validation Metrics:")
        print(f"- Accuracy:  {np.mean([m['accuracy'] for m in val_metrics]):.4f}")
        print(f"- F1:       {np.mean([m['f1'] for m in val_metrics]):.4f}")
        print(f"- Recall:   {np.mean([m['recall'] for m in val_metrics]):.4f}")
        print(f"- Precision:{np.mean([m['precision'] for m in val_metrics]):.4f}")
        print(f"- ROC AUC:  {np.mean([m['roc_auc'] for m in val_metrics]):.4f}")
    
    # Aggregate test metrics
    test_metrics = [m['test'] for m in all_metrics if m and m['test']]
    if test_metrics:
        print("\n🔍 Aggregate Test Metrics:")
        print(f"- Accuracy:  {np.mean([m['accuracy'] for m in test_metrics]):.4f}")
        print(f"- F1:       {np.mean([m['f1'] for m in test_metrics]):.4f}")
        print(f"- Recall:   {np.mean([m['recall'] for m in test_metrics]):.4f}")
        print(f"- Precision:{np.mean([m['precision'] for m in test_metrics]):.4f}")
        print(f"- ROC AUC:  {np.mean([m['roc_auc'] for m in test_metrics]):.4f}")
    
    # Generate comparison plots
    if val_metrics and test_metrics:
        plot_combined_metrics_trend(val_metrics, test_metrics)

def main():
    # Initialize monitoring and resource limits
    initialize_monitoring()
    Config.apply_cpu_limits()
    
    print(f"=== Starting Training (CPU Threads: {torch.get_num_threads()}) ===")
    log_resource_usage("Start of training")
    
    # Get partitions
    partition_files = get_partition_files()
    if not partition_files:
        raise RuntimeError("No valid partitions found")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_metrics = []
    
    for i, pfile in enumerate(partition_files):
        metrics = process_partition(
            pfile,
            device,
            i+1,
            len(partition_files)
        )
        if metrics:
            all_metrics.append(metrics)
    
    # Final reporting
    _final_aggregate_report(all_metrics)
    log_resource_usage("End")

if __name__ == "__main__":
    os.nice(19) 
    main()
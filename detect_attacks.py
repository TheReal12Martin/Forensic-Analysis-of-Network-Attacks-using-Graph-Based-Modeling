import argparse
from pcap_processor import PCAPProcessor
from classifier import NetworkAttackClassifier
from torch_geometric.data import Data
import torch
import time
import psutil
import os

def main():
    print("\n" + "="*60)
    print("=== NETWORK ATTACK DETECTION SYSTEM ===")
    print("="*60 + "\n")

    torch.set_num_threads(4)  # Limits CPU thread usage
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Set CUDA memory configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    print("[DEBUG] CUDA memory configuration set")
    
    parser = argparse.ArgumentParser(
        description="Network Attack Detection for Large PCAPs (Optimized for GTX 1650 Ti)")
    parser.add_argument("pcap_file", help="Path to PCAP file")
    parser.add_argument("--model", default="best_model.pt", help="Path to trained model")
    parser.add_argument("--max_packets", type=int, default=5_000_000,
                       help="Max packets to process (recommended: 5M for 10GB PCAPs)")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="GPU batch size (default: 1024 for 4GB VRAM)")
    args = parser.parse_args()

    print("[DEBUG] Parsed arguments:")
    print(f"  PCAP file: {args.pcap_file}")
    print(f"  Model: {args.model}")
    print(f"  Max packets: {args.max_packets}")
    print(f"  Batch size: {args.batch_size}")

    # Initialize CUDA context
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    
    if device.type == 'cuda':
        print("[DEBUG] Initializing CUDA context...")
        _ = torch.randn(1, device='cuda')  # Force CUDA initialization
        print(f"[DEBUG] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[DEBUG] CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    
    # Start monitoring
    start_time = time.time()
    proc = psutil.Process()
    print(f"\n🔍 Starting analysis of {os.path.basename(args.pcap_file)}")
    print(f"💻 System Resources:")
    print(f"  RAM: {psutil.virtual_memory().percent}% used")
    print(f"  CPU: {psutil.cpu_percent()}% used")
    print(f"  Device: {device}")

    try:
        # Step 1: Process PCAP
        print("\n=== PCAP PROCESSING ===")
        processor = PCAPProcessor()
        raw_graph = processor.process_pcap(args.pcap_file, args.max_packets)

        if not raw_graph or len(raw_graph['nodes']) < 2:
            print("❌ Not enough nodes for analysis")
            return
        
        # After processor.process_pcap()
        if raw_graph is None:
            print("❌ Failed to create graph from PCAP")
            return

        # Explicit dimension check
        if raw_graph['x'].shape[1] != 13:
            print(f"❌ Invalid feature dimension: {raw_graph['x'].shape[1]}")
            return


        # Step 2: Classify
        print("\n=== CLASSIFICATION ===")
        print(f"[DEBUG] Initializing classifier...")
        classifier = NetworkAttackClassifier(args.model, device=device, batch_size=args.batch_size)
        print(f"[DEBUG] Classifying graph with {len(raw_graph['nodes'])} nodes...")
        results = classifier.classify(Data(**raw_graph))
        
        # Output results
        print("\n🔴 DETECTED ATTACKS:")
        attack_count = sum(results['predictions'])
        for node, pred, prob in zip(results['nodes'], results['predictions'], results['probabilities']):
            if pred == 1:
                print(f"  {node:20} -> Confidence: {prob[pred]:.2%}")
        print(f"\n📊 Summary: {attack_count} attacks detected out of {len(results['nodes'])} nodes")
        
        # Visualization
        print("\n=== VISUALIZATION ===")
        classifier.save_for_d3(raw_graph, results, "data/graph.json")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU OUT OF MEMORY ERROR")
        print("Possible solutions:")
        print(f"  1. Reduce --batch_size (current: {args.batch_size})")
        print(f"  2. Lower --max_packets (current: {args.max_packets})")
        print(f"  3. Use a GPU with more memory")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
    finally:
        total_time = time.time() - start_time
        print("\n=== RESOURCE USAGE ===")
        print(f"⏱ Total time: {total_time:.2f}s")
        print(f"📈 Peak RAM usage: {proc.memory_info().rss / 1024**2:.2f} MB")
        if device.type == 'cuda':
            peak_gpu = torch.cuda.max_memory_allocated()/1024**3
            print(f"🎮 Peak GPU memory: {peak_gpu:.2f}GB")
        
        print("\n" + "="*60)
        print("=== ANALYSIS COMPLETE ===")
        print("="*60 + "\n")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
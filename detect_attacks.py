import argparse
from pcap_processor import PCAPProcessor
from classifier import NetworkAttackClassifier
from torch_geometric.data import Data
import torch
import time
import psutil
import os

def main():
    parser = argparse.ArgumentParser(
        description="Network Attack Detection for Large PCAPs (Optimized for GTX 1650 Ti)")
    parser.add_argument("pcap_file", help="Path to PCAP file")
    parser.add_argument("--model", default="best_model.pt", help="Path to trained model")
    parser.add_argument("--max_packets", type=int, default=5_000_000,
                       help="Max packets to process (recommended: 5M for 10GB PCAPs)")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="GPU batch size (default: 1024 for 4GB VRAM)")
    args = parser.parse_args()

    # Start monitoring
    start_time = time.time()
    proc = psutil.Process()
    print(f"\n🔍 Starting analysis of {os.path.basename(args.pcap_file)}")
    print(f"System RAM: {psutil.virtual_memory().percent}% used | CPU: {psutil.cpu_percent()}%")

    # Step 1: Process PCAP
    try:
        processor = PCAPProcessor()
        raw_graph = processor.process_pcap(args.pcap_file, args.max_packets)

        if not raw_graph or len(raw_graph['nodes']) < 2:
            print("❌ Not enough nodes for analysis")
            return

        # Step 2: Classify
        print(f"\n🧠 Classifying (Batch size: {args.batch_size})")
        classifier = NetworkAttackClassifier(args.model, batch_size=args.batch_size)
        results = classifier.classify(Data(**raw_graph))
        
        # Output results (only attacks)
        print("\n🔴 Detected Attacks:")
        attack_count = 0
        for node, pred, prob in zip(results['nodes'], results['predictions'], results['probabilities']):
            if pred == 1:
                print(f"{node:20} -> Confidence: {prob[pred]:.2%}")
                attack_count += 1
        print(f"\nTotal attacks detected: {attack_count}/{len(results['nodes'])}")
        
        # Visualization
        classifier.visualize_results(raw_graph, results)
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU Out of Memory! Try:")
        print("- Reduce --batch_size (e.g., 512)")
        print("- Lower --max_packets")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        print(f"\n⏱ Total time: {time.time() - start_time:.2f}s")
        print(f"Peak RAM usage: {proc.memory_info().rss / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
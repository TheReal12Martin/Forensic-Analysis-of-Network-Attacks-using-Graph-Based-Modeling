import argparse
from pcap_processor import PCAPProcessor
from classifier import NetworkAttackClassifier
from torch_geometric.data import Data
import torch

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Network Attack Detection from PCAP files")
    parser.add_argument(
        "pcap_file",
        help="Path to PCAP file to analyze")
    parser.add_argument(
        "--model",
        default="best_model.pt",
        help="Path to trained model")
    parser.add_argument(
        "--max_packets",
        type=int,
        default=None,
        help="Maximum number of packets to process")
    args = parser.parse_args()

    # Step 1: Process PCAP
    print("\nStarting PCAP processing...")
    processor = PCAPProcessor()
    raw_graph = processor.process_pcap(args.pcap_file, args.max_packets)

    if raw_graph is None:
        print("Failed to process PCAP file - no valid network traffic found")
        return
    
    # Verify we have enough nodes
    if len(raw_graph['nodes']) < 2:
        print(f"Not enough nodes ({len(raw_graph['nodes'])}) for meaningful analysis")
        return
    
    # Convert to PyG Data
    data = Data(
        x=raw_graph['x'],
        edge_index=raw_graph['edge_index'],
        edge_attr=raw_graph['edge_attr'],
        y=raw_graph['y']
    )
    
    # Verify feature dimensions
    num_features = data.x.shape[1]
    print(f"\nGraph contains {len(data.x)} nodes with {num_features} features each")
    
    # Step 2: Classify
    print("\nClassifying network nodes...")
    try:
        classifier = NetworkAttackClassifier(args.model, expected_features=num_features)
        results = classifier.classify(data)
        
        # Step 3: Output results
        print("\n🔍 Classification Results:")
        for node, pred, prob in zip(
            results['nodes'],
            results['predictions'],
            results['probabilities']
        ):
            status = "ATTACK" if pred == 1 else "Normal"
            print(f"{node:20} -> {status:8} (confidence: {prob[pred]:.2%})")
        
        # Generate visualization
        classifier.visualize_results(raw_graph, results)
        print(f"\nVisualization saved to attack_graph.html")
        
    except Exception as e:
        print(f"\nClassification failed: {str(e)}")

if __name__ == "__main__":
    main()
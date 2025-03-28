import sys
from train import train
from reality_checks import ModelValidator

def print_report(results):
    print("\n=== Final Results ===")
    print(f"Behavioral Leakage Score: {results['behavioral_leakage']['leakage_score']:.2%}")
    print(f"Test Accuracy: {results['performance']['accuracy']:.4f}")
    
    # Properly format classification report
    print("\nDetailed Classification Report:")
    report = results['performance']['report']
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for metric, score in value.items():
                print(f"  {metric}: {score:.4f}")
        else:
            print(f"{key}: {value:.4f}")

def main():
    try:
        print("=============== Starting Training ================")
        model, data, raw_data, node_mapping = train()
        
        print("\n=============== Running Validation ===============")
        validator = ModelValidator(model, raw_data, data, node_mapping)
        
        results = {
            'behavioral_leakage': validator.check_behavioral_leakage(),
            'performance': validator.evaluate_performance()
        }
        
        print_report(results)
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
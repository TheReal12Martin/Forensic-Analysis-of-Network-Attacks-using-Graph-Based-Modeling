import sys
from train import train
from reality_checks import ModelValidator

def main():
    try:
        print("=============== Starting Training ================")
        model, data, raw_data, ip_mapping = train()
        
        print("\n=============== Running Validation ===============")
        validator = ModelValidator(model, raw_data, data, ip_mapping)
        
        results = {
            'ip_leakage': validator.check_ip_leakage(),
            'performance': validator.evaluate_performance()
        }
        
        print("\n=== Validation Results ===")
        print(f"IP Leakage Score: {results['ip_leakage']['leakage_score']:.2%}")
        print(f"Test Accuracy: {results['performance']['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
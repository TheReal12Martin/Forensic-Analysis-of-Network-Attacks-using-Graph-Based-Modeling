from train import train
from reality_checks import ModelValidator
import sys
import json
from datetime import datetime

def main():
    try:
        print("=============== Starting Training ================")
        model, data, raw_data = train()
        
        print("\n=============== Running Validation ===============")
        validator = ModelValidator(model, raw_data, data)
        results = {
            'timestamp': datetime.now().isoformat(),
            'ip_leakage': validator.check_ip_leakage(),
            'feature_sensitivity': validator.check_feature_sensitivity()
        }
        
        with open('validation_report.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\nValidation completed successfully")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
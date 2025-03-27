import sys
from train import train
from reality_checks import ModelValidator

def main():
    try:
        print("=============== Starting Training ================")
        model, data, raw_data, ip_mapping = train()  # Now getting ip_mapping
        
        print("\n=============== Running Validation ===============")
        validator = ModelValidator(
            model=model,
            raw_data=raw_data,
            pyg_data=data,
            ip_mapping=ip_mapping  # Passing the mapping
        )
        
        results = {
            'ip_leakage': validator.check_ip_leakage(),
            'feature_sensitivity': validator.check_feature_sensitivity()
        }
        
        print("\nValidation Results:")
        print(f"IP Leakage Score: {results['ip_leakage']['leakage_score']:.2%}")
        print(f"Feature Sensitivity Delta: {results['feature_sensitivity']['delta']:.4f}")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
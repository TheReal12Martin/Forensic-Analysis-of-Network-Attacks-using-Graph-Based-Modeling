import sys
from train import train
from evaluate import evaluate

def main():
    try:
        print("=== Starting Training ===")
        model, data = train()
        
        print("\n=== Evaluating Model ===")
        evaluate(model, data)
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
from train import train
from evaluate import evaluate

if __name__ == "__main__":
    model, data = train()
    evaluate(model, data)
import torch
from train import train
from config import Config

def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')

if __name__ == "__main__":
    model, data = train()
    evaluate(model, data)
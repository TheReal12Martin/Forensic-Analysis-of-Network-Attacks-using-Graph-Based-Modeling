import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import train
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter

def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # Compute accuracy
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')

    # Check class distribution in test set
    print("Class distribution in test set:", Counter(data.y[data.test_mask].cpu().numpy()))

    # Confusion Matrix
    cm = confusion_matrix(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Classification Report
    print(classification_report(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), target_names=['Benign', 'Malicious'], zero_division=0))

if __name__ == "__main__":
    model, data = train()
    evaluate(model, data)
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_metrics_trend(val_metrics, test_metrics, output_file="metrics_trend.png"):
    """Plot validation vs test metric trends"""
    metrics = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        # Validation trend
        plt.plot(
            range(1, len(val_metrics)+1),
            [m[metric] for m in val_metrics],
            label='Validation',
            marker='o',
            color='blue'
        )
        
        # Test trend
        plt.plot(
            range(1, len(test_metrics)+1),
            [m[metric] for m in test_metrics],
            label='Test',
            marker='s',
            color='green'
        )
        
        plt.title(metric.upper())
        plt.xlabel('Partition')
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
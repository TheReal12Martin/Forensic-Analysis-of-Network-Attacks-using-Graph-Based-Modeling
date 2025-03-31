import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from config import Config

def evaluate_partition(model, data, partition_name=""):
    """Comprehensive evaluation with better error handling"""
    try:
        print(f"\n=== Evaluating Partition {partition_name} ===")
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            results = {}
            for mask_name, mask in [('train', data.train_mask), 
                                  ('val', data.val_mask), 
                                  ('test', data.test_mask)]:
                if mask.sum() == 0:
                    continue
                    
                # Calculate basic metrics
                correct = (pred[mask] == data.y[mask]).sum().item()
                total = mask.sum().item()
                acc = correct / total
                loss = F.cross_entropy(out[mask], data.y[mask]).item()
                
                # Calculate class-wise metrics if CLASS_NAMES exists
                report = ""
                confusion = ""
                try:
                    report = classification_report(
                        data.y[mask].cpu().numpy(),
                        pred[mask].cpu().numpy(),
                        target_names=Config.CLASS_NAMES,
                        zero_division=0
                    )
                    confusion = confusion_matrix(
                        data.y[mask].cpu().numpy(),
                        pred[mask].cpu().numpy()
                    )
                except AttributeError:
                    # Fallback if CLASS_NAMES not configured
                    report = classification_report(
                        data.y[mask].cpu().numpy(),
                        pred[mask].cpu().numpy(),
                        zero_division=0
                    )
                    confusion = confusion_matrix(
                        data.y[mask].cpu().numpy(),
                        pred[mask].cpu().numpy()
                    )
                
                results[mask_name] = {
                    'accuracy': acc,
                    'loss': loss,
                    'report': report,
                    'confusion': confusion
                }
        
        # Print results
        for split, metrics in results.items():
            print(f"\n{split.upper()} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Loss: {metrics['loss']:.4f}")
            print("\nClassification Report:")
            print(metrics['report'])
            print("\nConfusion Matrix:")
            print(metrics['confusion'])
        
        return results
        
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")
        raise
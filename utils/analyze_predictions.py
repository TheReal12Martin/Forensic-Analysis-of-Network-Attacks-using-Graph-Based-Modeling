import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def analyze_predictions(prediction_csv):
    """
    Analyzes the predictions CSV file and generates metrics and visualizations.

    Args:
        prediction_csv (str): Path to the predictions CSV file.
    """
    # Load the predictions CSV file
    df = pd.read_csv(prediction_csv)

    # Check if the required columns are present
    required_columns = ['label', 'prediction']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the predictions CSV file.")

    # Extract true labels and predictions
    true_labels = df['label']
    predictions = df['prediction']

    # Generate classification report
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Benign', 'Malicious']))

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix.png'.")

    # Calculate accuracy
    accuracy = (true_labels == predictions).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # Distribution of predictions
    plt.figure(figsize=(8, 6))
    sns.countplot(x='prediction', data=df, palette='Set2')
    plt.xticks([0, 1], ['Benign', 'Malicious'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions')
    plt.savefig('prediction_distribution.png')
    plt.close()
    print("Prediction distribution plot saved to 'prediction_distribution.png'.")

    # Summary of predictions
    print("\nSummary of Predictions:")
    print(df['prediction'].value_counts())

if __name__ == "__main__":
    # Path to the predictions CSV file
    prediction_csv = "data/preprocessed_features_predictions.csv"  # Replace with your file path

    # Analyze the predictions
    analyze_predictions(prediction_csv)
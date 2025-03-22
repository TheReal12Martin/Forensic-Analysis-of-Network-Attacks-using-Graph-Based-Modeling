import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def analyze_predictions(prediction_csv):
    """
    Analyzes the predictions CSV file and generates visualizations for the prediction distribution.

    Args:
        prediction_csv (str): Path to the predictions CSV file.
    """
    # Load the predictions CSV file
    df = pd.read_csv(prediction_csv)

    # Check if the required column is present
    if 'prediction' not in df.columns:
        raise ValueError("Column 'prediction' is missing from the predictions CSV file.")

    # Distribution of predictions
    plt.figure(figsize=(8, 6))
    sns.countplot(x='prediction', data=df, palette='Set2', hue='prediction', legend=False)
    plt.xticks([0, 1], ['Benign', 'Malicious'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions')
    plt.savefig('prediction_distribution.png')
    plt.close()
    logging.info("Prediction distribution plot saved to 'prediction_distribution.png'.")

    # Summary of predictions
    prediction_summary = df['prediction'].value_counts()
    logging.info("\nSummary of Predictions:")
    print(prediction_summary)

    # Save the summary to a text file
    with open("prediction_summary.txt", "w") as f:
        f.write("Summary of Predictions:\n")
        f.write(prediction_summary.to_string())

if __name__ == "__main__":
    # Path to the predictions CSV file
    prediction_csv = "/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018_predictions.csv"  # Replace with your file path

    # Analyze the predictions
    analyze_predictions(prediction_csv)
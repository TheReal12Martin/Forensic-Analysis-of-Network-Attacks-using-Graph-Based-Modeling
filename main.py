from train import train
from evaluate import evaluate
from utils.extract_features import extract_features
from utils.combine_csvs import combine_csvs
from utils.predict import predict_on_csv
from utils.analyze_predictions import analyze_predictions
from config import Config

def main():
    # Step 1: Train and evaluate the model
    print("Training and evaluating the model...")
    model, data = train()
    evaluate(model, data)

    # Step 2: Extract features from each PCAP file
    #print("Extracting features from PCAP files...")
    #extract_features()

    # Step 3: Combine the extracted CSV files into a single CSV
    #print("Combining CSV files...")
    #combine_csvs()

    # Step 4: Predict on the preprocessed data
    #print("Predicting on the preprocessed data...")
    #predict_on_csv("/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018.csv", "gat_network_security_model.pth")


    # Step 5: Analyze Predictions
    #print("Analyzing Predictions")
    #analyze_predictions("/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018_predictions.csv")



if __name__ == "__main__":
    main()
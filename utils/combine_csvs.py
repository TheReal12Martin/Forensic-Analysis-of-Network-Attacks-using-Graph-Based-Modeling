import pandas as pd
import os
def combine_csvs():
    # Directory containing CSV files
    csv_dir = '/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/CIC2018'

    # List all CSV files
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

    # Combine CSV files
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Save combined CSV
    combined_csv_path = '/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/combined_Friday02032018.csv'
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV saved to {combined_csv_path}")
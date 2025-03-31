#!/usr/bin/env python3
import pandas as pd
import os
from glob import glob
from config import Config
import argparse

def analyze_all_files(data_folder):
    """Check labels in all CSV files in a directory"""
    # Get all CSV files
    all_files = glob(os.path.join(data_folder, "*.csv"))
    
    if not all_files:
        print(f"❌ No CSV files found in {data_folder}")
        return

    print(f"\n🔍 Analyzing {len(all_files)} files in: {data_folder}")
    print("="*60)
    
    # Header
    print(f"{'File':<40} | {'Raw Labels':<30} | {'Mapped Labels':<15} | Issues")
    print("-"*100)

    for file in all_files:
        try:
            # Read first 1000 rows (no header inference)
            df = pd.read_csv(file, usecols=['label'], nrows=1000, header=0)
            
            # Standardize labels
            clean_labels = df['label'].astype(str).str.strip().str.title()
            raw_unique = df['label'].astype(str).unique()
            mapped_unique = clean_labels.map(Config.LABEL_MAPPING).unique()
            
            # Detect issues
            issues = []
            if len(mapped_unique) == 1 and mapped_unique[0] == 1 and "benign" in os.path.basename(file).lower():
                issues.append("⚠️ Benign file mapped as malicious")
            if any(clean_labels.isna()):
                issues.append("⚠️ Empty labels detected")
            unmapped = set(clean_labels) - set(Config.LABEL_MAPPING.keys())
            if unmapped:
                issues.append(f"⚠️ Unmapped labels: {unmapped}")
            
            # Print report
            print(f"{os.path.basename(file):<40} | {str(raw_unique[:3]):<30} | {str(mapped_unique):<15} | {', '.join(issues)}")
            
        except Exception as e:
            print(f"{os.path.basename(file):<40} | {'ERROR':<30} | {'-'*15} | ❌ {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check label mapping for all CSV files in a directory")
    parser.add_argument("--data-folder", type=str, required=True,
                       help="Path to folder containing CSV files (e.g. '/path/to/CSVs/')")
    args = parser.parse_args()
    
    print("\n📋 Current Label Mapping:")
    print({k: v for k, v in Config.LABEL_MAPPING.items() if v == 0}, "... (malicious mappings omitted)")
    
    analyze_all_files(args.data_folder)
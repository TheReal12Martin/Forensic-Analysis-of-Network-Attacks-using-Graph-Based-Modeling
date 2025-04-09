import pandas as pd
from itertools import zip_longest, chain
from tqdm import tqdm
import os
import csv

def interleave_large_csvs(file_paths, output_path, chunksize=100000):
    """
    Memory-efficient interleaving that preserves all data from both files.
    
    Args:
        file_paths (list): List of CSV file paths to interleave
        output_path (str): Output file path
        chunksize (int): Number of rows to process at a time (default: 100,000)
    """
    # First verify input files
    print("\nInput File Analysis:")
    for f in file_paths:
        size_gb = os.path.getsize(f) / (1024**3)
        with open(f) as temp:
            row_count = sum(1 for _ in temp) - 1
        print(f"• {f}\n  Rows: {row_count:,} | Size: {size_gb:.2f}GB")

    # Initialize readers with consistent dtypes
    readers = [pd.read_csv(f, chunksize=chunksize, dtype='object', low_memory=False) 
              for f in file_paths]

    # Get headers (assuming all files have same columns)
    headers = []
    for reader in readers:
        first_chunk = next(reader)
        headers.append(list(first_chunk.columns))
        readers = [chain([first_chunk], reader) for first_chunk, reader in zip([first_chunk], readers)]

    # Verify headers match
    if not all(h == headers[0] for h in headers):
        print("\nWarning: Header mismatch detected!")
        print("Using headers from first file:", headers[0])

    # Write output header
    pd.DataFrame(columns=headers[0]).to_csv(output_path, index=False)

    # Process chunks with size tracking
    total_rows_processed = 0
    with tqdm(desc="Processing", unit="rows") as pbar:
        while True:
            chunks = []
            try:
                chunks = [next(reader) for reader in readers]
            except StopIteration:
                break

            # Interleave all rows (preserve everything)
            interleaved = []
            max_len = max(len(c) for c in chunks)
            
            for i in range(max_len):
                for chunk in chunks:
                    if i < len(chunk):
                        interleaved.append(chunk.iloc[i])

            # Write to file
            if interleaved:
                pd.DataFrame(interleaved).to_csv(
                    output_path,
                    mode='a',
                    header=False,
                    index=False
                )
                total_rows_processed += len(interleaved)
                pbar.update(len(interleaved))

    # Final verification
    output_size = os.path.getsize(output_path) / (1024**3)
    print(f"\nProcess Complete\n{'='*40}")
    print(f"Total Rows Written: {total_rows_processed:,}")
    print(f"Output Size: {output_size:.2f}GB")
    print(f"Output saved to: {output_path}")

# Your file paths
csv_files = [
    '/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018/wednesday_21_02_2018_benign_part2.csv',
    '/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018/wednesday_21_02_2018_benign_part3.csv',
    '/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018/wednesday_21_02_2018_benign_part4.csv',
    '/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018/wednesday_21_02_2018_hoic_part1.csv'
]

interleave_large_csvs(
    csv_files,
    output_path='/home/martin/TFG/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling/CSVs/BCCC-CIC2018/wednesday_21_02_2018_merged.csv',
    chunksize=50000  # Conservative chunk size for large files
)
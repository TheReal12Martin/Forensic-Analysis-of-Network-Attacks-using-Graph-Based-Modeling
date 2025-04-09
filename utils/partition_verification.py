import numpy as np
import glob
import os
from tqdm import tqdm

def analyze_partitions(folder="data_partitions"):
    """Comprehensive partition analysis"""
    print("\n=== Partition Analysis ===")
    valid = 0
    single_class = 0
    small = 0
    
    for p_file in tqdm(sorted(glob.glob(f"{folder}/partition_*.npz"))):
        try:
            with np.load(p_file) as data:
                # Class distribution
                unique, counts = np.unique(data['labels'], return_counts=True)
                
                if len(unique) < 2:
                    single_class += 1
                    print(f"\n❌ {os.path.basename(p_file)}: Single class {unique[0]} ({counts[0]} samples)")
                elif len(data['labels']) < 1000:
                    small += 1
                    print(f"\n⚠️ {os.path.basename(p_file)}: Small partition ({len(data['labels'])} samples)")
                else:
                    valid += 1
                    print(f"\n✅ {os.path.basename(p_file)}: Classes {dict(zip(unique, counts))}")
                    
        except Exception as e:
            print(f"\n🔥 Corrupted {os.path.basename(p_file)}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"- Valid partitions: {valid}")
    print(f"- Single-class partitions: {single_class}")
    print(f"- Too small partitions: {small}")
    print(f"- Total files scanned: {valid + single_class + small}")

if __name__ == "__main__":
    analyze_partitions()
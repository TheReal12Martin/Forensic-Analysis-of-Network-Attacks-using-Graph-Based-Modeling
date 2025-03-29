import numpy as np
import glob
from tqdm import tqdm

def verify_partitions(folder="data_partitions"):
    """Verify all partition files are properly formatted"""
    print("\n=== Verifying Partitions ===")
    for p_file in tqdm(sorted(glob.glob(f"{folder}/partition_*.npz"))):
        try:
            print(f"\nChecking {p_file}...")
            with np.load(p_file, allow_pickle=True) as data:
                # Check required fields
                required = ['features', 'labels', 'src_ips', 'dst_ips']
                missing = [f for f in required if f not in data]
                if missing:
                    raise ValueError(f"Missing fields: {missing}")

                # Check data types
                print("Data types:")
                for k, v in data.items():
                    print(f"{k}: {v.dtype} (shape: {v.shape})")

                # Sample check
                print("\nSample data:")
                print("Src IPs:", data['src_ips'][:3])
                print("Dst IPs:", data['dst_ips'][:3])
                print("Features shape:", data['features'][0].shape)
                print("Labels:", np.unique(data['labels'], return_counts=True))

        except Exception as e:
            print(f"\nERROR in {p_file}: {str(e)}")
            raise

if __name__ == "__main__":
    verify_partitions()
import h5py
import sys

def inspect_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("Keys:", list(f.keys()))
        for key in f.keys():
            print(f"  Dataset: {key}, Shape: {f[key].shape}, Dtype: {f[key].dtype}")

if __name__ == "__main__":
    inspect_h5(sys.argv[1])

import h5py
import numpy as np

path = "D:/nosaka/data/train/10plots_128images_FalserandomMode_NumberFrom1.h5"
try:
    with h5py.File(path, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        if 'raw' in f:
            raw = f['raw'][:]
            print(f"RAW Shape: {raw.shape}")
            print(f"RAW Stats: Min={raw.min()}, Max={raw.max()}, Mean={raw.mean()}")
        if 'label' in f:
            lbl = f['label'][:]
            print(f"LABEL Shape: {lbl.shape}")
            print(f"LABEL Stats: Min={lbl.min()}, Max={lbl.max()}, Mean={lbl.mean()}")
except Exception as e:
    print(f"Error: {e}")

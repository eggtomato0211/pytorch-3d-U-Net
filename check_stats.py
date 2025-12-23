import h5py
import numpy as np

path = "D:/nosaka/data/data_multichannel_robust/train/sample_0000.h5"
with h5py.File(path, 'r') as f:
    raw = f['raw'][:]
    if raw.ndim == 4: raw = raw[0] # handle batch dim if present

    print(f"Min: {raw.min():.4f}, Max: {raw.max():.4f}")
    
    # plot_3d_sample.py logic
    thresh_v1 = raw.min() + (raw.max() - raw.min()) * 0.2
    
    # plot_3d.ps1 logic (default)
    thresh_v2 = np.percentile(raw, 99.5)
    
    print(f"Thresh V1 (20% range): {thresh_v1:.4f}")
    print(f"Thresh V2 (99.5% tile): {thresh_v2:.4f}")
    
    count_v1 = np.sum(raw > thresh_v1)
    count_v2 = np.sum(raw > thresh_v2)
    
    print(f"Points > V1: {count_v1} ({count_v1/raw.size*100:.2f}%)")
    print(f"Points > V2: {count_v2} ({count_v2/raw.size*100:.2f}%)")

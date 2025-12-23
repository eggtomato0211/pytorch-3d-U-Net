import h5py
path = 'D:/nosaka/outputs/predict_noisy/128images_10plots_fixed_randomFalse_NumberFrom1_predictions.h5'
with h5py.File(path, 'r') as f:
    print(f"File: {path}")
    print(f"Keys: {list(f.keys())}")
    for k in f.keys():
        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

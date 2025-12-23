import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_mig(input_path, output_path, save_path="prediction_result_mip.png"):
    print(f"Loading Input/Label: {input_path}")
    print(f"Loading Prediction: {output_path}")

    with h5py.File(input_path, 'r') as f_in:
        raw = f_in['raw'][:]
        label = f_in['label'][:]
    
    with h5py.File(output_path, 'r') as f_out:
        pred = f_out['predictions'][:]
        
    if pred.ndim == 4:
        pred = pred[0]

    # --- Statistics ---
    print(f"Raw   Min: {raw.min():.4f}, Max: {raw.max():.4f}, Mean: {raw.mean():.4f}")
    print(f"Label Min: {label.min():.4f}, Max: {label.max():.4f}, Mean: {label.mean():.4f}")
    print(f"Pred  Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}")

    # --- Maximum Intensity Projection (MIP) ---
    # Project along Z axis (depth)
    mip_raw = np.max(raw, axis=0) # ZYX -> YX
    mip_label = np.max(label, axis=0)
    mip_pred = np.max(pred, axis=0)
    
    # Use fixed scale based on Label for fair comparison
    vmin, vmax = 0, 1.0 # Assuming normalized label
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Input (MIP)
    ax = axes[0]
    im = ax.imshow(mip_raw, cmap='gray')
    ax.set_title("Input (MIP)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # 2. Prediction (MIP)
    ax = axes[1]
    # Show prediction with same scale as label to check absolute intensity
    im = ax.imshow(mip_pred, cmap='inferno') 
    ax.set_title(f"Prediction (MIP)\nMax: {mip_pred.max():.2f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # 3. Ground Truth (MIP)
    ax = axes[2]
    im = ax.imshow(mip_label, cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth (MIP)\n(0.0-1.0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # 4. Profile (Center Line) - Normalized Comparison
    ax = axes[3]
    mid_y = mip_label.shape[0] // 2
    
    # Normalize profiles for shape comparison
    prof_label = mip_label[mid_y, :]
    prof_pred = mip_pred[mid_y, :]
    
    prof_label_norm = (prof_label - prof_label.min()) / (prof_label.max() - prof_label.min() + 1e-9)
    prof_pred_norm = (prof_pred - prof_pred.min()) / (prof_pred.max() - prof_pred.min() + 1e-9)

    ax.plot(prof_label_norm, label='GT (Norm)', color='green', alpha=0.5)
    ax.plot(prof_pred_norm, label='Pred (Norm)', color='red', linestyle='--')
    ax.set_title("Normalized Profile (Shape Check)")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, default='project/configs/train_config.yaml')
    parser.add_argument('--check_file', type=str, help='Path to check stats for a single file')
    args = parser.parse_args()

    if args.check_file:
        print(f"Checking file: {args.check_file}")
        with h5py.File(args.check_file, 'r') as f:
            raw = f['raw'][:]
            label = f['label'][:]
            
            print(f"Raw shape: {raw.shape}")
            if raw.ndim == 4:
                print(f"Ch0 (TIE) - Min: {raw[0].min():.4f}, Max: {raw[0].max():.4f}, Mean: {raw[0].mean():.4f}")
                print(f"Ch1 (Raw) - Min: {raw[1].min():.4f}, Max: {raw[1].max():.4f}, Mean: {raw[1].mean():.4f}")
            else:
                print(f"Raw - Min: {raw.min():.4f}, Max: {raw.max():.4f}, Mean: {raw.mean():.4f}")
                
            print(f"Label shape: {label.shape}")
            print(f"Label - Min: {label.min():.4f}, Max: {label.max():.4f}, Mean: {label.mean():.4f}")
        exit()

    input_h5 = r"D:\nosaka\data\data_for_training\test\sample_0000.h5"
    output_h5 = r"D:\nosaka\outputs\predict\final\sample_0000_predictions.h5"
    
    visualize_mig(input_h5, output_h5)

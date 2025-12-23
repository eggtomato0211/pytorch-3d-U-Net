import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_3d(file_path):
    print(f"Loading: {file_path}")
    with h5py.File(file_path, 'r') as f:
        # (1, Z, Y, X) -> (Z, Y, X)
        raw_ds = f['raw']
        label_ds = f['label']
        
        # Handle shapes: (1, Z, Y, X) or (Z, Y, X)
        if raw_ds.ndim == 4 and raw_ds.shape[0] == 1:
            raw = raw_ds[0]
        else:
            raw = raw_ds[:]
            
        if label_ds.ndim == 4 and label_ds.shape[0] == 1:
            label = label_ds[0]
        else:
            label = label_ds[:]
        
    print(f"Raw shape: {raw.shape}, Max: {raw.max():.4f}, Min: {raw.min():.4f}")
    
    # Grid for plotting
    if len(raw.shape) == 3:
        Z, Y, X = raw.shape
    elif len(raw.shape) == 2:
        Y, X = raw.shape
        Z = 1
        raw = np.expand_dims(raw, axis=0)
    else:
        raise ValueError(f"Unknown shape: {raw.shape}")
    
    fig = plt.figure(figsize=(18, 10))
    
    # --- 1. 3D Scatter Plot (Raw) ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # Use a threshold to visualize "structures" -> e.g. top 10% intensity or absolute threshold
    threshold = raw.min() + (raw.max() - raw.min()) * 0.2
    
    z, y, x = np.where(raw > threshold)
    v = raw[raw > threshold]
    
    # Downsample if too many points (scatter3d is slow)
    if len(z) > 5000:
        idx = np.random.choice(len(z), 5000, replace=False)
        z, y, x, v = z[idx], y[idx], x[idx], v[idx]
        
    img1 = ax1.scatter(x, y, z, c=v, cmap='jet', alpha=0.3, s=2)
    ax1.set_title(f"Raw Input > {threshold:.2f} (Scatter)")
    ax1.set_xlim(0, X); ax1.set_ylim(0, Y); ax1.set_zlim(0, Z)
    
    # --- 2. 3D Scatter Plot (Label) ---
    ax2 = fig.add_subplot(2, 3, 4, projection='3d')
    if label.max() > 0:
        thresh_lbl = label.max() * 0.5
        if len(label.shape) == 3:
            z, y, x = np.where(label > thresh_lbl)
            v = label[label > thresh_lbl]
        else: # 2D
             y, x = np.where(label > thresh_lbl)
             z = np.zeros_like(x)
             v = label[label > thresh_lbl]
        ax2.scatter(x, y, z, c='r', alpha=1.0, s=10)
    ax2.set_title("Label (Beads Pos)")
    ax2.set_xlim(0, X); ax2.set_ylim(0, Y); ax2.set_zlim(0, Z)

    # --- 3. MIP Projections (XYZ) ---
    # XY Plane (Top view)
    ax3 = fig.add_subplot(2, 3, 2)
    ax3.imshow(np.max(raw, axis=0), cmap='gray')
    ax3.set_title("MIP: XY Plane (Top)")
    
    # XZ Plane (Side view)
    ax4 = fig.add_subplot(2, 3, 3)
    ax4.imshow(np.max(raw, axis=1), cmap='gray', aspect='auto')
    ax4.set_title("MIP: XZ Plane (Side)")

    # Center Slice (XY) - to see noise texture
    ax5 = fig.add_subplot(2, 3, 5)
    center_z = Z // 2
    ax5.imshow(raw[center_z, :, :], cmap='gray')
    ax5.set_title(f"Cross Section: Z={center_z}")
    
    # Hist (Intensity Distribution)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(raw.flatten(), bins=100, log=True)
    ax6.set_title("Intensity Histogram (Log)")
    
    plt.tight_layout()
    output_png = "check_3d_plot.png"
    plt.savefig(output_png)
    print(f"Saved visualization to: {os.path.abspath(output_png)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to h5 file')
    args = parser.parse_args()
    plot_3d(args.file)

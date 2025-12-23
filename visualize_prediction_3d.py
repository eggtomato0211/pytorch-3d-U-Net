import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_comparison_3d(test_file_path, prediction_file_path, output_path=None):
    """
    Visualize raw, label, and prediction in 3D scatter plots side by side.
    
    Args:
        test_file_path: Path to test H5 file containing raw and label
        prediction_file_path: Path to prediction H5 file containing predictions
        output_path: Optional output path for the PNG file
    """
    print(f"Loading test data: {test_file_path}")
    print(f"Loading prediction: {prediction_file_path}")
    
    # Load test data (raw and label)
    with h5py.File(test_file_path, 'r') as f:
        print(f"Test file keys: {list(f.keys())}")
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
    
    # Load prediction
    with h5py.File(prediction_file_path, 'r') as f:
        print(f"Prediction file keys: {list(f.keys())}")
        pred_ds = f['predictions']
        
        # Handle shapes: (1, Z, Y, X) or (Z, Y, X)
        if pred_ds.ndim == 4 and pred_ds.shape[0] == 1:
            prediction = pred_ds[0]
        else:
            prediction = pred_ds[:]
    
    print(f"Raw shape: {raw.shape}, Range: [{raw.min():.4f}, {raw.max():.4f}]")
    print(f"Label shape: {label.shape}, Range: [{label.min():.4f}, {label.max():.4f}]")
    print(f"Prediction shape: {prediction.shape}, Range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    # Get dimensions
    if len(raw.shape) == 3:
        Z, Y, X = raw.shape
    else:
        raise ValueError(f"Expected 3D data, got shape: {raw.shape}")
    
    # Create figure with 3 rows and 3 columns
    fig = plt.figure(figsize=(20, 18))
    
    # ========== ROW 1: 3D Scatter Plots ==========
    
    # --- 1. Raw Input (3D Scatter) ---
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    threshold_raw = raw.min() + (raw.max() - raw.min()) * 0.2
    z, y, x = np.where(raw > threshold_raw)
    v = raw[raw > threshold_raw]
    
    # Downsample if too many points
    if len(z) > 5000:
        idx = np.random.choice(len(z), 5000, replace=False)
        z, y, x, v = z[idx], y[idx], x[idx], v[idx]
    
    ax1.scatter(x, y, z, c=v, cmap='jet', alpha=0.3, s=2)
    ax1.set_title(f"Raw Input (Scatter, threshold={threshold_raw:.2f})", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, X); ax1.set_ylim(0, Y); ax1.set_zlim(0, Z)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # --- 2. Label (3D Scatter) ---
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    if label.max() > 0:
        thresh_lbl = label.max() * 0.5
        z, y, x = np.where(label > thresh_lbl)
        v = label[label > thresh_lbl]
        
        # Downsample if needed
        if len(z) > 5000:
            idx = np.random.choice(len(z), 5000, replace=False)
            z, y, x, v = z[idx], y[idx], x[idx], v[idx]
        
        ax2.scatter(x, y, z, c=v, cmap='Reds', alpha=0.8, s=10)
    ax2.set_title("Label (Ground Truth)", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, X); ax2.set_ylim(0, Y); ax2.set_zlim(0, Z)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # --- 3. Prediction (3D Scatter) ---
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    if prediction.max() > 0:
        thresh_pred = prediction.max() * 0.5
        z, y, x = np.where(prediction > thresh_pred)
        v = prediction[prediction > thresh_pred]
        
        # Downsample if needed
        if len(z) > 5000:
            idx = np.random.choice(len(z), 5000, replace=False)
            z, y, x, v = z[idx], y[idx], x[idx], v[idx]
        
        ax3.scatter(x, y, z, c=v, cmap='Greens', alpha=0.8, s=10)
    ax3.set_title("Prediction", fontsize=12, fontweight='bold')
    ax3.set_xlim(0, X); ax3.set_ylim(0, Y); ax3.set_zlim(0, Z)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    
    # ========== ROW 2: MIP XY Projections (Top View) ==========
    
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(np.max(raw, axis=0), cmap='gray')
    ax4.set_title("Raw - MIP XY (Top View)", fontsize=11)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(np.max(label, axis=0), cmap='hot')
    ax5.set_title("Label - MIP XY (Top View)", fontsize=11)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(np.max(prediction, axis=0), cmap='hot')
    ax6.set_title("Prediction - MIP XY (Top View)", fontsize=11)
    ax6.axis('off')
    
    # ========== ROW 3: MIP XZ Projections (Side View) ==========
    
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(np.max(raw, axis=1), cmap='gray', aspect='auto')
    ax7.set_title("Raw - MIP XZ (Side View)", fontsize=11)
    ax7.axis('off')
    
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(np.max(label, axis=1), cmap='hot', aspect='auto')
    ax8.set_title("Label - MIP XZ (Side View)", fontsize=11)
    ax8.axis('off')
    
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(np.max(prediction, axis=1), cmap='hot', aspect='auto')
    ax9.set_title("Prediction - MIP XZ (Side View)", fontsize=11)
    ax9.axis('off')
    
    plt.tight_layout()
    
    # Save output
    if output_path is None:
        # Generate output name from prediction file
        pred_basename = os.path.basename(prediction_file_path).replace('_predictions.h5', '')
        output_path = f"visualization_{pred_basename}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {os.path.abspath(output_path)}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize raw, label, and prediction in 3D')
    parser.add_argument('test_file', type=str, help='Path to test H5 file (contains raw and label)')
    parser.add_argument('prediction_file', type=str, help='Path to prediction H5 file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output PNG path (optional)')
    
    args = parser.parse_args()
    plot_comparison_3d(args.test_file, args.prediction_file, args.output)

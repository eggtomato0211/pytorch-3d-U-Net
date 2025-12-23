import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def load_dataset(f, key):
    """Load dataset and handle different shapes (1, Z, Y, X) or (Z, Y, X)"""
    ds = f[key]
    if ds.ndim == 4 and ds.shape[0] == 1:
        return ds[0]
    else:
        return ds[:]

def plot_3d_single(data, title, ax_scatter, ax_mip_xy, ax_mip_xz, cmap='viridis'):
    """Plot a single 3D dataset with scatter plot and MIP projections."""
    if len(data.shape) != 3:
        print(f"Warning: Expected 3D data, got shape {data.shape}. Skipping.")
        return
    
    Z, Y, X = data.shape
    print(f"  {title}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
    
    # --- 3D Scatter Plot ---
    if data.max() > data.min():
        threshold = data.min() + (data.max() - data.min()) * 0.3
        z, y, x = np.where(data > threshold)
        v = data[data > threshold]
        
        if len(z) > 5000:
            idx = np.random.choice(len(z), 5000, replace=False)
            z, y, x, v = z[idx], y[idx], x[idx], v[idx]
        
        if len(z) > 0:
            ax_scatter.scatter(x, y, z, c=v, cmap=cmap, alpha=0.8, s=5)
    
    ax_scatter.set_title(f"{title}", fontsize=12, fontweight='bold', pad=10)
    ax_scatter.set_xlim(0, X)
    ax_scatter.set_ylim(0, Y)
    ax_scatter.set_zlim(0, Z)
    ax_scatter.grid(False)
    ax_scatter.set_facecolor('white')
    
    # --- MIP XY (Top View) ---
    mip_xy = np.max(data, axis=0)
    ax_mip_xy.imshow(mip_xy, cmap=cmap)
    ax_mip_xy.set_title(f"{title}", fontsize=12, fontweight='bold', pad=10)
    ax_mip_xy.axis('off')
    
    # --- MIP XZ (Side View) ---
    mip_xz = np.max(data, axis=1)
    ax_mip_xz.imshow(mip_xz, cmap=cmap, aspect='auto')
    ax_mip_xz.set_title(f"{title}", fontsize=12, fontweight='bold', pad=10)
    ax_mip_xz.axis('off')

def compare_test_and_prediction(test_h5, pred_h5, output_path=None):
    """
    Compare test data (raw, label) with prediction results using 3D scatter plots only.
    
    Args:
        test_h5: Path to test H5 file (contains raw and label)
        pred_h5: Path to prediction H5 file (contains predictions)
        output_path: Optional output path for the PNG file
    """
    print(f"\n{'='*60}")
    print(f"Loading test data: {test_h5}")
    print(f"Loading prediction: {pred_h5}")
    print(f"{'='*60}\n")
    
    # Load test data
    with h5py.File(test_h5, 'r') as f:
        raw = load_dataset(f, 'raw')
        label = load_dataset(f, 'label')
    
    # Load prediction
    with h5py.File(pred_h5, 'r') as f:
        prediction = load_dataset(f, 'predictions')
    
    # Create figure with only 3D scatter plots
    fig = plt.figure(figsize=(18, 6), facecolor='white')
    
    # Helper function to plot 3D scatter
    def plot_3d_scatter(data, ax, title, cmap, threshold_ratio=0.2):
        if len(data.shape) != 3:
            return
        
        Z, Y, X = data.shape
        print(f"  {title}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
        
        if data.max() > data.min():
            threshold = data.min() + (data.max() - data.min()) * threshold_ratio
            z, y, x = np.where(data > threshold)
            v = data[data > threshold]
            
            # Downsample if too many points
            if len(z) > 8000:
                idx = np.random.choice(len(z), 8000, replace=False)
                z, y, x, v = z[idx], y[idx], x[idx], v[idx]
            
            if len(z) > 0:
                scatter = ax.scatter(x, y, z, c=v, cmap=cmap, alpha=0.9, s=8, edgecolors='none')
                plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        
        ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_zlim(0, Z)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
    
    # Column 1: Raw (cyan colormap)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_3d_scatter(raw, ax1, 'Raw', 'cool', threshold_ratio=0.2)
    
    # Column 2: Label (hot colormap - red/yellow)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_3d_scatter(label, ax2, 'Label', 'hot', threshold_ratio=0.5)
    
    # Column 3: Prediction (viridis colormap - blue/green/yellow)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_3d_scatter(prediction, ax3, 'Prediction', 'viridis', threshold_ratio=0.5)
    
    plt.tight_layout(pad=3.0)
    
    # Save output - default to D drive
    if output_path is None:
        output_dir = "D:\\visualizations"
        os.makedirs(output_dir, exist_ok=True)
        pred_basename = os.path.basename(pred_h5).replace('_predictions.h5', '')
        output_path = os.path.join(output_dir, f"comparison_{pred_basename}.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved comparison: {os.path.abspath(output_path)}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare test data (raw, label) with prediction results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_prediction.py \\
    "D:/nosaka/data/3d-holography_output/Test/128images_1plots_fixed_randomFalse_NumberFrom1.h5" \\
    "D:/nosaka/outputs/holography/128images_1plots_fixed_randomFalse_NumberFrom1_predictions.h5"
        """
    )
    
    parser.add_argument('test_file', type=str, help='Path to test H5 file (contains raw and label)')
    parser.add_argument('prediction_file', type=str, help='Path to prediction H5 file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output PNG path (optional)')
    
    args = parser.parse_args()
    
    compare_test_and_prediction(args.test_file, args.prediction_file, args.output)

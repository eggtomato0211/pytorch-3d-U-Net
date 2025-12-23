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

def plot_3d_scatter(data, title, ax, cmap='viridis', threshold_ratio=0.3):
    """
    Plot a single 3D dataset with scatter plot only.
    
    Args:
        data: 3D numpy array (Z, Y, X)
        title: Title for the plot
        ax: Matplotlib 3D axis for scatter plot
        cmap: Colormap to use
        threshold_ratio: Ratio for thresholding (0-1)
    """
    if len(data.shape) != 3:
        print(f"Warning: Expected 3D data, got shape {data.shape}. Skipping.")
        return
    
    Z, Y, X = data.shape
    print(f"  {title}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
    
    # --- 3D Scatter Plot ---
    if data.max() > data.min():
        # Use threshold to show structure
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

def visualize_h5_file(h5_path, output_mode='combined', output_dir=None):
    """
    Visualize all datasets in an H5 file with 3D plots.
    
    Args:
        h5_path: Path to H5 file
        output_mode: 'combined' (all in one image) or 'separate' (one file per dataset)
        output_dir: Output directory (default: same as h5 file)
    """
    print(f"\n{'='*60}")
    print(f"Loading: {h5_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return
    
    # Open H5 file and find all datasets
    with h5py.File(h5_path, 'r') as f:
        # Get all dataset keys
        dataset_keys = []
        
        def find_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_keys.append(name)
        
        f.visititems(find_datasets)
        
        print(f"\nFound {len(dataset_keys)} dataset(s): {dataset_keys}")
        
        if len(dataset_keys) == 0:
            print("No datasets found in file!")
            return
        
        # Load all datasets
        datasets = {}
        for key in dataset_keys:
            try:
                datasets[key] = load_dataset(f, key)
                print(f"Loaded '{key}': shape={datasets[key].shape}")
            except Exception as e:
                print(f"Warning: Could not load '{key}': {e}")
        
        if len(datasets) == 0:
            print("No valid datasets to visualize!")
            return
    
    # Prepare output directory - default to D drive
    if output_dir is None:
        output_dir = "D:\\visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    
    # Colormap and threshold assignment
    colormap_dict = {
        'raw': 'cool',           # cyan/blue
        'label': 'hot',          # red/yellow
        'predictions': 'viridis' # blue/green/yellow
    }
    
    threshold_dict = {
        'raw': 0.2,
        'label': 0.5,
        'predictions': 0.5
    }
    
    # --- MODE 1: Combined visualization (3D scatter only) ---
    if output_mode == 'combined':
        num_datasets = len(datasets)
        fig = plt.figure(figsize=(6 * num_datasets, 6), facecolor='white')
        
        print(f"\nGenerating combined visualization...")
        
        for idx, (key, data) in enumerate(datasets.items()):
            # Create single row of 3D scatter plots
            ax = fig.add_subplot(1, num_datasets, idx + 1, projection='3d')
            
            # Choose colormap and threshold
            cmap = colormap_dict.get(key, 'viridis')
            threshold_ratio = threshold_dict.get(key, 0.3)
            plot_3d_scatter(data, key, ax, cmap=cmap, threshold_ratio=threshold_ratio)
        
        plt.tight_layout(pad=3.0)
        output_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved combined visualization: {output_path}")
        plt.close()
    
    # --- MODE 2: Separate files (3D scatter only) ---
    elif output_mode == 'separate':
        print(f"\nGenerating separate visualizations...")
        
        for key, data in datasets.items():
            fig = plt.figure(figsize=(8, 6), facecolor='white')
            
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            # Choose colormap and threshold
            cmap = colormap_dict.get(key, 'viridis')
            threshold_ratio = threshold_dict.get(key, 0.3)
            plot_3d_scatter(data, key, ax, cmap=cmap, threshold_ratio=threshold_ratio)
            
            plt.tight_layout(pad=3.0)
            safe_key = key.replace('/', '_')
            output_path = os.path.join(output_dir, f"{base_name}_{safe_key}.png")
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}")
            plt.close()
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize all datasets in an H5 file with 3D plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combined visualization (all datasets in one image)
  python visualize_h5_3d.py data.h5
  
  # Separate files for each dataset
  python visualize_h5_3d.py data.h5 --mode separate
  
  # Specify output directory
  python visualize_h5_3d.py data.h5 --output ./visualizations
        """
    )
    
    parser.add_argument('h5_file', type=str, help='Path to H5 file')
    parser.add_argument('--mode', '-m', type=str, default='combined',
                        choices=['combined', 'separate'],
                        help='Output mode: "combined" (all in one image) or "separate" (one file per dataset)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: same as input file)')
    
    args = parser.parse_args()
    
    visualize_h5_file(args.h5_file, output_mode=args.mode, output_dir=args.output)

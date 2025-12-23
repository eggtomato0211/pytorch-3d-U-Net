import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# --- Configuration ---
RAW_DIR = Path(r"D:/nosaka/data/3d-holography_output/Test_misty")
PRED_DIR = Path(r"D:/nosaka/outputs/predict_noisy")
OUTPUT_PNG = Path(r"c:/Users/Owner/mizusaki/pytorch-3dunet/cs-13/prediction_results_3d.png")

# Choose a sample
BASENAME = "128images_10plots_fixed_randomFalse_NumberFrom1"
RAW_FILE = RAW_DIR / f"{BASENAME}_misty.h5"
PRED_FILE = PRED_DIR / f"{BASENAME}_predictions.h5"

def plot_split_colors(ax, vol, min_val, boundary_val, max_points, show_haze=True):
    """
    データを「霧(青)」と「信号(黄)」に分けてプロットする関数 (check_misty_final.pngと同一仕様)
    """
    # 1. 霧パート (Blue/Cyan) - 弱い信号
    if show_haze:
        z1, y1, x1 = np.where((vol >= min_val) & (vol < boundary_val))
        if len(z1) > 0:
            if len(z1) > max_points:
                idx = np.random.choice(len(z1), max_points, replace=False)
                z1, y1, x1 = z1[idx], y1[idx], x1[idx]
            vals1 = vol[z1, y1, x1]
            norm1 = (vals1 - min_val) / (boundary_val - min_val + 1e-6)
            colors1 = plt.cm.winter(norm1)
            colors1[:, 3] = 0.2 + 0.6 * norm1 
            ax.scatter(x1, y1, z1, c=colors1, s=2, linewidth=0)

    # 2. 信号パート (Red/Yellow) - 強い信号
    z2, y2, x2 = np.where(vol >= boundary_val)
    if len(z2) > 0:
        if len(z2) > max_points // 2:
            idx = np.random.choice(len(z2), max_points // 2, replace=False)
            z2, y2, x2 = z2[idx], y2[idx], x2[idx]
        vals2 = vol[z2, y2, x2]
        norm2 = (vals2 - boundary_val) / (vol.max() - boundary_val + 1e-6)
        colors2 = plt.cm.hot(norm2)
        colors2[:, 3] = 0.8 + 0.2 * norm2 
        ax.scatter(x2, y2, z2, c=colors2, s=2, linewidth=0)

def load_data():
    print(f"Loading Raw/Label from: {RAW_FILE.name}")
    with h5py.File(RAW_FILE, "r") as f:
        raw = f["raw"][:]
        label = f["label"][:]
    
    print(f"Loading Prediction from: {PRED_FILE.name}")
    with h5py.File(PRED_FILE, "r") as f:
        pred = f["predictions"][:]
        # Pred is often (C, Z, Y, X), we want (Z, Y, X)
        if pred.ndim == 4:
            pred = pred[0]
            
    return raw, label, pred

def plot_3d(raw, label, pred):
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f"3D Comparison: {BASENAME}\n(Raw Misty vs Label vs Prediction)", fontsize=16, fontweight='bold')

    # Thresholds (from visualize_noisy_3d.py)
    data_mean = np.mean(raw)
    signal_boundary = data_mean * 2.0
    mist_min = data_mean * 0.1
    max_pts = 50000

    # --- 1. Raw ---
    ax1 = fig.add_subplot(131, projection='3d')
    plot_split_colors(ax1, raw, min_val=mist_min, boundary_val=signal_boundary, max_points=max_pts, show_haze=True)
    ax1.set_title('Raw (Input)', fontsize=14, fontweight='bold')

    # --- 2. Label ---
    ax2 = fig.add_subplot(132, projection='3d')
    z_l, y_l, x_l = np.where(label > 0.5)
    ax2.scatter(x_l, y_l, z_l, c='red', s=2, alpha=0.8)
    ax2.set_title('Ground Truth Label', fontsize=14, fontweight='bold')

    # --- 3. Prediction (Original Scale, Cleaned) ---
    ax3 = fig.add_subplot(133, projection='3d')
    # Use fixed threshold for original 0-1 scale
    p_boundary = 0.1 
    plot_split_colors(ax3, pred, min_val=0, boundary_val=p_boundary, max_points=max_pts, show_haze=False)
    ax3.set_title('Model Prediction', fontsize=14, fontweight='bold')

    # Common Settings
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 128); ax.set_ylim(0, 128); ax.set_zlim(0, 128)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=-30)
        ax.invert_zaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"✅ 3D Plot saved to: {OUTPUT_PNG}")

def plot_mip(raw, label, pred):
    # Maximum Intensity Projection for clearer overview
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"MIP Comparison: {BASENAME}", fontsize=16, fontweight='bold')

    data_list = [raw, label, pred]
    titles = ["Raw", "Label", "Prediction"]
    cmaps = ["hot", "hot", "hot"] 

    for i, (data, title, cmap) in enumerate(zip(data_list, titles, cmaps)):
        # Z-Projection (Top View)
        axes[i, 0].imshow(np.max(data, axis=0), cmap=cmap)
        axes[i, 0].set_title(f"{title} - MIP(Z)")
        axes[i, 0].axis('off')

        # Y-Projection (Side View 1)
        axes[i, 1].imshow(np.max(data, axis=1), cmap=cmap)
        axes[i, 1].set_title(f"{title} - MIP(Y)")
        axes[i, 1].axis('off')

        # X-Projection (Side View 2)
        axes[i, 2].imshow(np.max(data, axis=2), cmap=cmap)
        axes[i, 2].set_title(f"{title} - MIP(X)")
        axes[i, 2].axis('off')

    mip_output = OUTPUT_PNG.parent / "prediction_results_mip.png"
    plt.tight_layout()
    plt.savefig(mip_output, dpi=150, bbox_inches='tight')
    print(f"✅ MIP Plot saved to: {mip_output}")

def main():
    if not RAW_FILE.exists():
        print(f"Error: {RAW_FILE} not found")
        return
    if not PRED_FILE.exists():
        print(f"Error: {PRED_FILE} not found")
        return

    raw, label, pred = load_data()
    plot_3d(raw, label, pred)
    plot_mip(raw, label, pred)

if __name__ == "__main__":
    main()

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# パス設定
original_file = Path(r"D:/nosaka/data/3d-holography_output/Train/128images_10plots_fixed_randomFalse_NumberFrom1025.h5")
composite_file = Path(r"D:/nosaka/data/3d-holography_output/Train_composite_test/128images_10plots_fixed_randomFalse_NumberFrom1025_composite.h5")

print(f"📂 比較対象:")
print(f"  元データ: {original_file.name}")
print(f"  複合ノイズデータ: {composite_file.name}")

# データ読み込み
with h5py.File(original_file, "r") as f:
    original_raw = f["raw"][:]
    
with h5py.File(composite_file, "r") as f:
    composite_raw = f["raw"][:]

# --- 1. MIP (Maximum Intensity Projection) 比較 ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Composite Noise Model: Original vs Composite\n(Haze + DC Offset + Shot/Read Noise)", 
              fontsize=14, fontweight='bold')

# 元データのMIP
axes[0, 0].imshow(np.max(original_raw, axis=0), cmap='hot')
axes[0, 0].set_title('Original - MIP (Top View)', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(np.max(original_raw, axis=1), cmap='hot')
axes[0, 1].set_title('Original - MIP (Side View 1)', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(np.max(original_raw, axis=2), cmap='hot')
axes[0, 2].set_title('Original - MIP (Side View 2)', fontsize=11)
axes[0, 2].axis('off')

# 複合ノイズデータのMIP
axes[1, 0].imshow(np.max(composite_raw, axis=0), cmap='hot')
axes[1, 0].set_title('Composite - MIP (Top View)', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(np.max(composite_raw, axis=1), cmap='hot')
axes[1, 1].set_title('Composite - MIP (Side View 1)', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(np.max(composite_raw, axis=2), cmap='hot')
axes[1, 2].set_title('Composite - MIP (Side View 2)', fontsize=11)
axes[1, 2].axis('off')

plt.tight_layout()
mip_output_path = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\composite_comparison_mip.png")
plt.savefig(mip_output_path, dpi=150, bbox_inches='tight')
print(f"✅ MIPプロット保存: {mip_output_path}")

# --- 2. 3D Scatter Plot 比較 ---
fig3d = plt.figure(figsize=(18, 8))
fig3d.suptitle(f"3D Visualization: Composite Noise Comparison", fontsize=14, fontweight='bold')

# しきい値以上のポイントを抽出
threshold_orig = np.percentile(original_raw, 99.5)
threshold_comp = np.percentile(composite_raw, 99.5)

z_orig, y_orig, x_orig = np.where(original_raw > threshold_orig)
z_comp, y_comp, x_comp = np.where(composite_raw > threshold_comp)

intensity_orig = original_raw[z_orig, y_orig, x_orig]
intensity_comp = composite_raw[z_comp, y_comp, x_comp]

# 元データの3Dプロット
ax1 = fig3d.add_subplot(121, projection='3d')
ax1.scatter(x_orig, y_orig, z_orig, c=intensity_orig, cmap='hot', s=0.5, alpha=0.3)
ax1.set_title('Original Data', fontsize=12)
ax1.set_xlim(0, 128); ax1.set_ylim(0, 128); ax1.set_zlim(0, 128)

# 複合ノイズデータの3Dプロット
ax2 = fig3d.add_subplot(122, projection='3d')
ax2.scatter(x_comp, y_comp, z_comp, c=intensity_comp, cmap='hot', s=0.5, alpha=0.3)
ax2.set_title('Composite Noisy Data', fontsize=12)
ax2.set_xlim(0, 128); ax2.set_ylim(0, 128); ax2.set_zlim(0, 128)

plt.tight_layout()
scatter_output_path = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\composite_comparison_3d.png")
plt.savefig(scatter_output_path, dpi=150, bbox_inches='tight')
print(f"✅ 3Dプロット保存: {scatter_output_path}")

print("\n🎉 全可視化完了！")

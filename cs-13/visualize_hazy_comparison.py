import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# パス設定 (Hazyテスト用)
original_file = Path(r"D:/nosaka/data/3d-holography_output/Train/128images_10plots_fixed_randomFalse_NumberFrom1025.h5")
hazy_file = Path(r"D:/nosaka/data/3d-holography_output/Train_hazy_test/128images_10plots_fixed_randomFalse_NumberFrom1025_hazy.h5")

print(f"📂 比較対象:")
print(f"  元データ: {original_file.name}")
print(f"  Hazyデータ: {hazy_file.name}")

# データ読み込み
with h5py.File(original_file, "r") as f:
    original_raw = f["raw"][:]
    
with h5py.File(hazy_file, "r") as f:
    hazy_raw = f["raw"][:]

# MIP (Maximum Intensity Projection)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Hazy Effect Comparison: Original vs Hazy (Foggy Clouds)\n{original_file.name}", 
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

# HazyデータのMIP
axes[1, 0].imshow(np.max(hazy_raw, axis=0), cmap='hot')
axes[1, 0].set_title('Hazy - MIP (Top View)', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(np.max(hazy_raw, axis=1), cmap='hot')
axes[1, 1].set_title('Hazy - MIP (Side View 1)', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(np.max(hazy_raw, axis=2), cmap='hot')
axes[1, 2].set_title('Hazy - MIP (Side View 2)', fontsize=11)
axes[1, 2].axis('off')

plt.tight_layout()

# MIP保存
output_path = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\hazy_comparison_mip.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ MIPプロット保存: {output_path}")

# Slice comparison
z_mid = original_raw.shape[0] // 2
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle(f"Hazy Effect: Slice Comparison", fontsize=14, fontweight='bold')

z_slices = [z_mid // 2, z_mid, z_mid + z_mid // 2]
for i, z in enumerate(z_slices):
    axes2[0, i].imshow(original_raw[z], cmap='gray')
    axes2[0, i].set_title(f"Original Slice (Z={z})")
    axes2[0, i].axis('off')
    
    axes2[1, i].imshow(hazy_raw[z], cmap='gray')
    axes2[1, i].set_title(f"Hazy Slice (Z={z})")
    axes2[1, i].axis('off')

plt.tight_layout()
slice_output_path = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\hazy_comparison_slices.png")
plt.savefig(slice_output_path, dpi=150, bbox_inches='tight')
print(f"✅ スプロット保存: {slice_output_path}")

print("\n🎉 可視化完了！")

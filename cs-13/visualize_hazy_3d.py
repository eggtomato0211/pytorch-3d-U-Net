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

# 3Dプロット作成
fig = plt.figure(figsize=(18, 8))
fig.suptitle(f"3D Visualization: Hazy Effect Comparison (Foggy Clouds)\n{original_file.name}", 
             fontsize=14, fontweight='bold')

# しきい値以上のポイントを抽出（上位1%の輝度の点）
threshold_orig = np.percentile(original_raw, 99)
threshold_hazy = np.percentile(hazy_raw, 99)

z_orig, y_orig, x_orig = np.where(original_raw > threshold_orig)
z_hazy, y_hazy, x_hazy = np.where(hazy_raw > threshold_hazy)

intensity_orig = original_raw[z_orig, y_orig, x_orig]
intensity_hazy = hazy_raw[z_hazy, y_hazy, x_hazy]

# 元データの3Dプロット
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(x_orig, y_orig, z_orig, 
                       c=intensity_orig, cmap='hot', 
                       s=1, alpha=0.3, vmin=original_raw.min(), vmax=original_raw.max())
ax1.set_title('Original Data', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 128); ax1.set_ylim(0, 128); ax1.set_zlim(0, 128)
plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)

# Hazyデータの3Dプロット
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(x_hazy, y_hazy, z_hazy, 
                       c=intensity_hazy, cmap='hot', 
                       s=1, alpha=0.3, vmin=hazy_raw.min(), vmax=hazy_raw.max())
ax2.set_title('Hazy Data (Cloudy Effect)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 128); ax2.set_ylim(0, 128); ax2.set_zlim(0, 128)
plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=5)

# 視点の設定
ax1.view_init(elev=20, azim=45)
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
output_path = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\hazy_comparison_3d.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 3Dプロット保存: {output_path}")

print("\n🎉 可視化完了！")

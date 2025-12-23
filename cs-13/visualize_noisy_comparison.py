import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# パス設定
original_dir = Path(r"D:/nosaka/data/3d-holography_output/Train")
noisy_dir = Path(r"D:/nosaka/data/3d-holography_output/Train_noisy")

# 処理済みのファイルを探す
noisy_files = list(noisy_dir.glob("*_noisy.h5"))
if not noisy_files:
    print("⚠ ノイズ付きファイルが見つかりません")
    exit(1)

# 最初のファイルを使用
noisy_file = noisy_files[0]
original_name = noisy_file.stem.replace("_noisy", "") + ".h5"
original_file = original_dir / original_name

print(f"📂 比較対象:")
print(f"  元データ: {original_file.name}")
print(f"  ノイズ付き: {noisy_file.name}")

# データ読み込み
with h5py.File(original_file, "r") as f:
    original_raw = f["raw"][:]
    
with h5py.File(noisy_file, "r") as f:
    noisy_raw = f["raw"][:]

print(f"\n📊 データ形状: {original_raw.shape}")
print(f"  元データ範囲: [{original_raw.min():.2f}, {original_raw.max():.2f}]")
print(f"  ノイズ付き範囲: [{noisy_raw.min():.2f}, {noisy_raw.max():.2f}]")

# 中央のスライスを選択
z_mid = original_raw.shape[0] // 2

# プロット作成
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"ノイズ付加の比較 (PSNR≈18dB)\n{original_file.name}", fontsize=14, fontweight='bold')

# 3つの異なるZ位置でプロット
z_positions = [z_mid // 2, z_mid, z_mid + z_mid // 2]

for idx, z in enumerate(z_positions):
    # 元データ
    ax_orig = axes[0, idx]
    im_orig = ax_orig.imshow(original_raw[z], cmap='gray', vmin=original_raw.min(), vmax=original_raw.max())
    ax_orig.set_title(f"元データ (Z={z})", fontsize=12)
    ax_orig.axis('off')
    plt.colorbar(im_orig, ax=ax_orig, fraction=0.046)
    
    # ノイズ付きデータ
    ax_noisy = axes[1, idx]
    im_noisy = ax_noisy.imshow(noisy_raw[z], cmap='gray', vmin=noisy_raw.min(), vmax=noisy_raw.max())
    ax_noisy.set_title(f"ノイズ付き (Z={z})", fontsize=12)
    ax_noisy.axis('off')
    plt.colorbar(im_noisy, ax=ax_noisy, fraction=0.046)

plt.tight_layout()

# 保存
output_path = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\noisy_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ プロット保存: {output_path}")

# 差分の統計情報
diff = np.abs(original_raw - noisy_raw)
print(f"\n📈 ノイズ統計:")
print(f"  平均絶対誤差: {diff.mean():.4f}")
print(f"  最大絶対誤差: {diff.max():.4f}")
print(f"  標準偏差: {diff.std():.4f}")

plt.show()

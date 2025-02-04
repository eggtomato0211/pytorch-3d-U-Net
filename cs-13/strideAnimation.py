import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 3Dデータのサイズ
volume_size = (128, 128, 128)
patch_shape = (128, 128, 128)

# ストライドの設定（16と64）
stride_values = [16, 64]

# アニメーション用のデータ
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# スライスのリストを生成する関数
def generate_slices(volume_shape, patch_shape, stride):
    slices = []
    for z in range(0, volume_shape[0] - patch_shape[0] + 1, stride):
        for y in range(0, volume_shape[1] - patch_shape[1] + 1, stride):
            for x in range(0, volume_shape[2] - patch_shape[2] + 1, stride):
                slices.append((z, y, x))
    return slices

# 各ストライドのスライスリストを生成
slices_16 = generate_slices(volume_size, patch_shape, stride_values[0])
slices_64 = generate_slices(volume_size, patch_shape, stride_values[1])

# アニメーション用のフレーム数
num_frames = max(len(slices_16), len(slices_64))

# アニメーション用の関数
def update(frame):
    axes[0].cla()
    axes[1].cla()
    
    axes[0].set_title(f"Stride = {stride_values[0]} (Frame {frame})")
    axes[1].set_title(f"Stride = {stride_values[1]} (Frame {frame})")

    # 背景のボリュームを描画
    volume = np.zeros(volume_size[:2])
    axes[0].imshow(volume, cmap="gray", alpha=0.3)
    axes[1].imshow(volume, cmap="gray", alpha=0.3)

    # スライスの枠を描画
    if frame < len(slices_16):
        z, y, x = slices_16[frame]
        rect = plt.Rectangle((x, y), patch_shape[1], patch_shape[2], edgecolor="blue", facecolor="none", lw=2)
        axes[0].add_patch(rect)

    if frame < len(slices_64):
        z, y, x = slices_64[frame]
        rect = plt.Rectangle((x, y), patch_shape[1], patch_shape[2], edgecolor="red", facecolor="none", lw=2)
        axes[1].add_patch(rect)

    axes[0].set_xlim(0, volume_size[1])
    axes[0].set_ylim(0, volume_size[2])
    axes[1].set_xlim(0, volume_size[1])
    axes[1].set_ylim(0, volume_size[2])

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, repeat=True)

# アニメーションを表示
plt.show()

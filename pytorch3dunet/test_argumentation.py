import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import gaussian_filter, gaussian_filter1d, shift

# ==============================================================
# ノイズ生成関数群
# ==============================================================

def add_poisson_gaussian(x, peak=120, read_std=0.008):
    """Poisson(ショットノイズ) + Gaussian(読み出しノイズ)"""
    x = x.astype(np.float32)
    vmin, vmax = float(x.min()), float(x.max())
    if vmax == vmin:
        return x.copy()
    x01 = (x - vmin) / (vmax - vmin)
    lam = np.clip(x01 * float(peak), 0.0, None)
    y01 = np.random.poisson(lam).astype(np.float32) / float(peak)
    y01 = y01 + np.random.normal(0.0, read_std, size=x.shape).astype(np.float32)
    y01 = np.clip(y01, 0.0, 1.0)
    return y01 * (vmax - vmin) + vmin

def add_bias_field(x, amp=0.04, sigma=30):
    """低周波ゆらぎ（背景ドリフト）"""
    x = x.astype(np.float32)
    noise = np.random.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=sigma)
    smooth /= np.max(np.abs(smooth)) + 1e-8
    return x + amp * smooth * (x.max() - x.min())

def add_correlated_gaussian(x, std=0.012, blur_sigma=1.8):
    """空間相関を持つガウスノイズ"""
    x = x.astype(np.float32)
    n = np.random.normal(0.0, std, size=x.shape).astype(np.float32)
    n = gaussian_filter(n, sigma=blur_sigma)
    return x + n

def add_z_streak(x, amp=0.03, f=3, phase=None):
    """z方向周期ストリーク（TIE特有の擬似）"""
    x = x.astype(np.float32)
    D, _, _ = x.shape
    z = np.arange(D, dtype=np.float32)
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)
    pattern = np.sin(2*np.pi * f * z / max(D, 1) + phase).astype(np.float32)
    pattern = pattern[:, None, None]
    return x + amp * pattern * (x.max() - x.min())

def gaussian_blur_along_z(x, sigma=0.8):
    """z方向ガウスぼけ"""
    x = x.astype(np.float32)
    return gaussian_filter1d(x, sigma=sigma, axis=0, mode='nearest')

def add_z_blur_blend(x, alpha=0.1, sigma=0.8):
    """z方向ぼけ混合（PSF変動模倣）"""
    blur = gaussian_blur_along_z(x, sigma=sigma)
    return (1.0 - alpha) * x + alpha * blur

def add_slice_jitter(x, max_shift=1.0, prob=0.3):
    """スライスごとのランダム平行移動"""
    x = x.astype(np.float32)
    D, H, W = x.shape
    y = x.copy()
    for z in range(D):
        if np.random.rand() < prob:
            dy = np.random.uniform(-max_shift, max_shift)
            dx = np.random.uniform(-max_shift, max_shift)
            y[z] = shift(y[z], shift=(dy, dx), order=1, mode='nearest', prefilter=False)
    return y


# ==============================================================
# PSNR調整関数
# ==============================================================

def compute_psnr(gt, noisy):
    gt = gt.astype(np.float32)
    noisy = noisy.astype(np.float32)
    data_range = float(gt.max() - gt.min()) or 1.0
    return peak_signal_noise_ratio(gt, noisy, data_range=data_range)

def tune_noise_strength(func, x, target_psnr, param_name, init_value, step=0.08, max_iter=20, **kwargs):
    """PSNRがtarget_psnrに近づくよう自動調整"""
    value = init_value
    best_y, best_psnr = None, None

    for i in range(max_iter):
        kwargs[param_name] = value
        y = func(x, **kwargs)
        psnr = compute_psnr(x, y)
        diff = abs(psnr - target_psnr)
        print(f"[{i+1:02d}] {func.__name__}: {param_name}={value:.4f} → PSNR={psnr:.2f} dB")

        if diff < 0.3:
            best_y, best_psnr = y, psnr
            break

        # PSNRが高い＝ノイズ弱い → 強める方向
        if psnr > target_psnr:
            value *= (1 + step)
        else:
            value *= (1 - step)

        best_y, best_psnr = y, psnr

    print(f"✅ {func.__name__} 完了: {param_name}≈{value:.4f}, PSNR≈{best_psnr:.2f} dB\n")
    return best_y, best_psnr


# ==============================================================
# 可視化関数
# ==============================================================

def plot_2d_and_3d(raw, noisy, title, save_prefix="output"):
    """rawとnoisyを比較プロット（2Dスライス＋3D点群）"""
    # 2Dスライスの中央断面
    z = raw.shape[0] // 2
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(raw[z], cmap='gray')
    plt.title('Raw')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy[z], cmap='gray')
    plt.title(f'Noisy ({title})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{title}_slices.png", dpi=150)
    plt.close()

    # 3Dスキャッタープロット
    threshold = np.percentile(raw, 99.5)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    z_idx, y_idx, x_idx = np.where(raw > threshold)
    ax.scatter(x_idx, y_idx, z_idx, c='blue', s=3, alpha=0.3)
    ax.set_title('Raw (3D)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.invert_zaxis()

    ax2 = fig.add_subplot(122, projection='3d')
    z_idx, y_idx, x_idx = np.where(noisy > threshold)
    ax2.scatter(x_idx, y_idx, z_idx, c='red', s=3, alpha=0.3)
    ax2.set_title(f'Noisy ({title})')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.invert_zaxis()

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{title}_3d.png", dpi=150)
    plt.close()


# ==============================================================
# メイン実行
# ==============================================================

if __name__ == "__main__":
    # --- TIE再構成rawデータの読み込み ---
    FILE_PATH = r"C:\Users\Owner\Desktop\test250\Number1.h5"
    with h5py.File(FILE_PATH, "r") as f:
        raw = f["raw"][:]  # rawデータセットを読み込み

    TARGET_PSNR = 25.0  # ← 目標PSNRを設定

    # --- 各ノイズを個別にPSNR調整 & 可視化 ---
    noise_tasks = [
        (add_poisson_gaussian, "peak", 60, dict(read_std=0.01)),
        (add_bias_field, "amp", 0.05, dict(sigma=25)),
        (add_correlated_gaussian, "std", 0.10, dict(blur_sigma=1.5)),
        (add_z_streak, "amp", 0.05, dict(f=3)),
        (add_z_blur_blend, "alpha", 0.25, dict(sigma=0.8)),
        (add_slice_jitter, "max_shift", 2.0, dict(prob=0.3)),
    ]

    for func, param, init, extra in noise_tasks:
        print("=" * 60)
        noisy, psnr = tune_noise_strength(func, raw, TARGET_PSNR, param, init, **extra)
        plot_2d_and_3d(raw, noisy, title=func.__name__)
        print(f"🎯 {func.__name__} → 最終PSNR={psnr:.2f} dB\n")

import os
import yaml
import torch
import h5py
import numpy as np
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio
from pytorch3dunet.unet3d.model import get_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import ndimage
from datetime import datetime
import random

# =============================
# ユーザ設定（ご指定どおり）
# =============================
CHECKPOINT_NOISY = r"D:\nosaka\checkpoint\best_checkpoint.pytorch"
CHECKPOINT_CLEAN = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128\patch=128_stride=32_fm=64_valpatch=128\best_checkpoint.pytorch"
TEST_FOLDER      = r"c:\Users\Owner\Desktop\test250"
CONFIG_PATH      = r"C:\Users\Owner\mizusaki\pytorch-3dunet\project\configs\train_config.yaml"

# 保存先（Dドライブの適当なフォルダ：日時でサブフォルダを作成）
SAVE_ROOT = r"D:\nosaka\predict_plots"
RUN_DIR   = os.path.join(SAVE_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
# =============================


def load_model_and_fix_keys(checkpoint_path, config):
    model = get_model(config['model'])
    is_data_parallel = torch.cuda.device_count() > 1
    if is_data_parallel:
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict_from_file = checkpoint['model_state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict_from_file.items():
        if is_data_parallel and not k.startswith('module.'):
            name = 'module.' + k
        elif not is_data_parallel and k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    return model


def compute_psnr(gt, pred):
    data_range = float(gt.max() - gt.min()) or 1.0
    return peak_signal_noise_ratio(
        gt.astype(np.float64), pred.astype(np.float64), data_range=data_range
    )


def calculate_sharpness(data):
    lap = ndimage.laplace(data.astype(np.float32))
    return float(np.var(lap))


def normalize_to_minus_one_one(data):
    min_val, max_val = data.min(), data.max()
    if max_val > min_val:
        return (data - min_val) / (max_val - min_val) * 2 - 1, min_val, max_val
    return data.astype(float), min_val, max_val


def denormalize_from_minus_one_one(data, min_val, max_val):
    if max_val > min_val:
        return (data + 1) / 2 * (max_val - min_val) + min_val
    return data.astype(float)


def run_inference(model, raw):
    norm, minv, maxv = normalize_to_minus_one_one(raw)
    t = torch.tensor(norm[None, None, ...], dtype=torch.float32)
    if torch.cuda.is_available():
        t = t.cuda()
    with torch.no_grad():
        out = model(t)
    pred = denormalize_from_minus_one_one(out.cpu().numpy().squeeze(), minv, maxv)
    return pred


def plot_depth_slices(raw, pred, label, save_path, title):
    depth = raw.shape[0]
    slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes[0, 0].set_title('Raw (Input)')
    axes[0, 1].set_title(title)
    axes[0, 2].set_title('Ground Truth')

    for i, idx in enumerate(slice_indices):
        axes[i, 0].imshow(raw[idx], cmap='gray')
        axes[i, 1].imshow(pred[idx], cmap='gray')
        axes[i, 2].imshow(label[idx], cmap='gray')
        axes[i, 0].set_ylabel(f"Depth: {idx}")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ 保存: {save_path}")


def plot_3d_top_percent(data, save_path, title, percentile=99.5, max_points=60000, seed=42, color='C0'):
    """
    体積データの上位パーセンタイルの点群を3D散布図で描画。
    点数が多すぎる場合は subsample して保存。
    """
    thr = np.percentile(data, percentile)
    z, y, x = np.where(data > thr)

    pts = np.stack([x, y, z], axis=1)
    n = pts.shape[0]
    if n == 0:
        print(f"⚠ 上位{percentile}%に相当する点が見つかりませんでした: {title}")
        # 空でも枠だけ保存
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title + " (no points)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✅ 保存(空図): {save_path}")
        return

    if n > max_points:
        random.seed(seed)
        idx = np.random.choice(n, size=max_points, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.3, c=color)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.invert_zaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ 保存: {save_path}")


def main():
    # 保存先フォルダ作成（Dドライブ）
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"🗂️ 保存先: {os.path.abspath(RUN_DIR)}")

    # 設定ファイル（モデル定義に使用）
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # モデル2本ロード
    print("🔄 モデル(ノイズ学習)読み込み中...")
    model_noisy = load_model_and_fix_keys(CHECKPOINT_NOISY, config)
    print("✅ モデル(ノイズ学習)OK")

    print("🔄 モデル(クリーン学習)読み込み中...")
    model_clean = load_model_and_fix_keys(CHECKPOINT_CLEAN, config)
    print("✅ モデル(クリーン学習)OK")

    # テスト対象列挙：_prediction を含まない .h5
    if not os.path.isdir(TEST_FOLDER):
        print(f"❌ テストフォルダが見つかりません: {TEST_FOLDER}")
        return

    h5_files = [
        os.path.join(TEST_FOLDER, f)
        for f in os.listdir(TEST_FOLDER)
        if f.lower().endswith('.h5') and "_prediction" not in f.lower()
    ]
    h5_files.sort()
    print(f"📂 対象ファイル数: {len(h5_files)}")
    if not h5_files:
        return

    for file_path in h5_files:
        base = os.path.splitext(os.path.basename(file_path))[0]
        print("\n" + "="*60)
        print(f"📄 処理中: {base}")

        with h5py.File(file_path, "r") as f:
            raw = f["raw"][:]
            label = f["label"][:]

        # 推論
        pred_noisy = run_inference(model_noisy, raw)
        pred_clean = run_inference(model_clean, raw)

        # ログ（端末表示のみ）
        print(f"PSNR noisy={compute_psnr(label, pred_noisy):.2f} / clean={compute_psnr(label, pred_clean):.2f}")
        print(f"Sharp noisy={calculate_sharpness(pred_noisy):.4f} / clean={calculate_sharpness(pred_clean):.4f}")

        # ===== 保存（2D & 3D） =====
        # 2Dスライス
        out_noisy_2d = os.path.join(RUN_DIR, f"{base}_noisyModel_slices.png")
        out_clean_2d = os.path.join(RUN_DIR, f"{base}_cleanModel_slices.png")
        plot_depth_slices(raw, pred_noisy, label, out_noisy_2d, "Denoised (Noisy-Trained Model)")
        plot_depth_slices(raw, pred_clean, label, out_clean_2d, "Denoised (Clean-Trained Model)")

        # 3Dプロット（上位パーセンタイルの点群）
        out_noisy_3d = os.path.join(RUN_DIR, f"{base}_noisyModel_3d.png")
        out_clean_3d = os.path.join(RUN_DIR, f"{base}_cleanModel_3d.png")
        plot_3d_top_percent(pred_noisy, out_noisy_3d, "3D Scatter (Noisy-Trained Model)", color='C2')
        plot_3d_top_percent(pred_clean, out_clean_3d, "3D Scatter (Clean-Trained Model)", color='C1')

    print("\n🎯 全ファイル完了！")
    print(f"📁 出力フォルダ: {RUN_DIR}")


if __name__ == "__main__":
    main()

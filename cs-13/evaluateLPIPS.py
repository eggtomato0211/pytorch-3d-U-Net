import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import lpips
import os
import numpy as np
from pytorch3dunet.unet3d.model import get_model
import yaml

# ==== 設定 ====
hdf_path = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\test250\Number40.h5"  # HDFファイルのパス
prediction_hdf_path = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\predictions\patch=128_stride=16_fm=64_valpatch=128\Number40_predictions.h5"  # 予測結果のHDFファイルのパス
checkpoint_path = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128\patch=128_stride=16_fm=64_valpatch=128\best_checkpoint.pytorch"

config_path = r'C:\Users\Owner\mizusaki\pytorch-3dunet\resources\3DUnet_denoising\train_config_regression.yaml'  # 設定ファイルのパス


# ==== HDFファイルからデータを読み込む関数 ====
def load_hdf_data(hdf_path, dataset_key):
    with h5py.File(hdf_path, "r") as f:
        data = np.array(f[dataset_key])
    return data

# ==== デバイス設定 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==== データの読み込み ====
# ここではデータセットのキー名を "label", "raw", "predictions" と仮定
label_data = load_hdf_data(hdf_path, "label")         # 例：shape = (D, H, W)
raw_data = load_hdf_data(hdf_path, "raw")               # 例：shape = (D, H, W)
predictions_data = load_hdf_data(prediction_hdf_path, "predictions")
predictions_data = predictions_data.squeeze()  # 不要な次元の削除

print(f"Label Data shape: {label_data.shape}")
print(f"Raw Data shape: {raw_data.shape}")
print(f"Predictions Data shape: {predictions_data.shape}")

# ==== 3Dボリュームをtorch.Tensorに変換 ==== 
# ※ここでは各データが (D, H, W) の形状であると仮定
label_volume = torch.tensor(label_data, dtype=torch.float32)
raw_volume = torch.tensor(raw_data, dtype=torch.float32)
pred_volume = torch.tensor(predictions_data, dtype=torch.float32)

# ==== LPIPSの準備 ==== 
# 'alex' や 'squeeze' など、好みのネットワークを選択可能（2D画像用のネットワーク）
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()  # 評価モード

# ==== 3Dボリュームに対して各スライスごとにLPIPSを計算する関数 ====
def compute_lpips_3d(volume1, volume2, lpips_fn, device, slice_axis=0):
    """
    volume1, volume2: torch.Tensor, shape = (D, H, W)
    slice_axis: どの軸でスライスするか（0, 1, 2）
    """
    # z軸方向に沿ってスライス
    num_slices = volume1.shape[slice_axis]
    lpips_total = 0.0
    count = 0

    for i in range(num_slices):
        # スライスを抽出（sliceごとに shape (H, W) となるように）
        if slice_axis == 0:
            slice1 = volume1[i, :, :]
            slice2 = volume2[i, :, :]
        elif slice_axis == 1:
            slice1 = volume1[:, i, :]
            slice2 = volume2[:, i, :]
        elif slice_axis == 2:
            slice1 = volume1[:, :, i]
            slice2 = volume2[:, :, i]
        else:
            raise ValueError("slice_axis は 0, 1, 2 のいずれかである必要があります。")

        # 形状を (1, 1, H, W) に変換 → さらに3チャンネルに拡張
        slice1 = slice1.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        slice2 = slice2.unsqueeze(0).unsqueeze(0)
        slice1 = slice1.repeat(1, 3, 1, 1).to(device)  # (1,3,H,W)
        slice2 = slice2.repeat(1, 3, 1, 1).to(device)

        # もし入力が [0, 1] の範囲なら、[-1, 1] に正規化
        slice1 = slice1 * 2 - 1
        slice2 = slice2 * 2 - 1

        with torch.no_grad():
            lpips_val = lpips_fn(slice1, slice2)
        lpips_total += lpips_val.item()
        count += 1

    return lpips_total / count if count > 0 else None

# ==== LPIPSの計算 ==== 
# 例として、D軸（axis=0）に沿って各スライスごとのLPIPSを平均

lpips_value_raw = compute_lpips_3d(label_volume, raw_volume, lpips_fn, device, slice_axis=0)
lpips_value_pred = compute_lpips_3d(label_volume, pred_volume, lpips_fn, device, slice_axis=0)

print(f"LPIPS (Raw vs. Label): {lpips_value_raw:.4f}")
print(f"LPIPS (Prediction vs. Label): {lpips_value_pred:.4f}")

import os
import yaml
import subprocess
import time
import torch
import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.predictor import StandardPredictor

# ===============================
# デバッグ用関数
# ===============================
def debug_stats(data, name):
    """データの shape、min、max、mean を出力"""
    print(f"[DEBUG] {name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")

def normalize_data(data):
    """最小値と最大値を使って正規化し、-1～1の範囲に収める"""
    min_value = data.min()
    max_value = data.max()
    return (data - min_value) / (max_value - min_value) * 2 - 1

def compute_psnr(label, prediction):
    """PSNR を計算（skimage の peak_signal_noise_ratio を利用）"""
    data_range = label.max() - label.min()
    return peak_signal_noise_ratio(label, prediction, data_range=data_range)

# -------------------------------
# モデルロード・推論用関数
# -------------------------------
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = {
        "name": "UNet3D",
        "in_channels": 1,
        "out_channels": 1,
        "f_maps": [16, 32, 64, 128, 256],
        "layer_order": "gcr",
        "num_groups": 8,
        "is_segmentation": False
    }
    model = get_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        label_data = f['label'][:]
        raw_data   = f['raw'][:]
    return label_data, raw_data

def run_validation_inference(model, raw_data):
    """eval() で推論を実行"""
    # raw_data を入力を確認
    print(f"[DEBUG] Raw Data: shape={raw_data.shape}, min={raw_data.min():.4f}, max={raw_data.max():.4f}, mean={raw_data.mean():.4f}")

    with torch.no_grad():
        input_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        debug_stats(input_tensor.cpu().numpy(), "Validation Input Tensor")
        prediction = model(input_tensor)
        debug_stats(prediction.cpu().numpy(), "Validation Raw Prediction")
    return prediction.squeeze().numpy()

def run_predict_inference(model, raw_data):
    """StandardPredictor 経由の推論（predict.py と同等の処理）"""
    predictor = StandardPredictor(model, output_dir=None, out_channels=1)
    with torch.no_grad():
        input_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        debug_stats(input_tensor.cpu().numpy(), "Predict Input Tensor")
        prediction = predictor.model(input_tensor)
        debug_stats(prediction.cpu().numpy(), "Predict Raw Prediction")
    return prediction.squeeze().numpy()

# ===============================
# メイン処理
# ===============================
def main():
    start_time = time.time()

    # --- 各種パス設定 ---
    #configファイルの設定を読み込む
    hdf5_file_folder = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\valSimple"
    #フォルダ内のhdf5ファイルを読み込む(endwithでhdf5ファイルを指定)
    h5_files = [os.path.join(hdf5_file_folder, f) for f in os.listdir(hdf5_file_folder) if f.endswith('.h5') and not f.endswith('_predictions.h5')]
    h5_file = h5_files[0]
    print(f"[DEBUG] HDF5 ファイル: {h5_file}")

    model_checkpoint = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128\patch=64_stride=48_fm=16_valpatch=128\best_checkpoint.pytorch"

    # --- モデル・データの読み込み ---
    print(">> モデルの読み込み")
    model = load_model(model_checkpoint)
    print(">> HDF5 データの読み込み")
    label_data, raw_data = load_hdf5_data(h5_file)

    debug_stats(label_data, "Label data (Raw from HDF5)")
    debug_stats(raw_data, "Raw data (Raw from HDF5)")

    # --- 正規化 ---
    label_data = normalize_data(label_data)
    raw_data   = normalize_data(raw_data)

    # --- PSNR: Raw と Label の比較 ---
    raw_psnr = compute_psnr(label_data, raw_data)
    print(f"[DEBUG] 【Raw と Label の PSNR】: {raw_psnr:.2f}")

    # ==============================
    # ① 直接推論（Validation 時）
    # ==============================
    print("\n>> バリデーション時のモデル推論を実行")
    prediction_validation = run_validation_inference(model, raw_data)
    # prediction_validation = normalize_data(prediction_validation)
    debug_stats(prediction_validation, "Validation Prediction after Normalize")
    psnr_validation = compute_psnr(label_data, prediction_validation)
    print(f"[DEBUG] バリデーション時の PSNR: {psnr_validation:.2f}")

    # # ==============================
    # # ② StandardPredictor 経由の推論 (predict.py と同じ処理)
    # # ==============================
    # print("\n>> StandardPredictor 経由の推論を実行")
    # prediction_predict = run_predict_inference(model, raw_data)
    # prediction_predict = normalize_data(prediction_predict)
    # debug_stats(prediction_predict, "Predict.py Prediction after Normalize")
    # psnr_predict = compute_psnr(label_data, prediction_predict)
    # print(f"[DEBUG] predict.py 実行時の PSNR: {psnr_predict:.2f}")

    # # ==============================
    # # ③ YAML 経由の predict3dunet 推論
    # # ==============================
    # print("\n>> YAML 経由の predict3dunet コマンドによる推論準備")
    # yaml_path = create_yaml_config(original_config, base_yaml_dir, model_name, test_file)
    
    # try:
    #     run_predict3dunet(yaml_path)
    # except Exception as e:
    #     print("[DEBUG] predict3dunet の実行でエラーが発生しました。出力ファイルの有無を確認してください。")
    #     print("エラー内容:", e)
    
    # prediction_file = test_file.replace(".h5", "_predictions.h5")
    # if not os.path.exists(prediction_file):
    #     print(f"[DEBUG] 予測結果ファイルが見つかりません: {prediction_file}")
    # else:
    #     prediction_pipeline = load_prediction_from_h5(prediction_file)
    #     prediction_pipeline = normalize_data(prediction_pipeline)
    #     debug_stats(prediction_pipeline, "Pipeline Prediction after Normalize")
    #     psnr_pipeline = compute_psnr(label_data, prediction_pipeline)
    #     print(f"[DEBUG] predict3dunet 経由の PSNR: {psnr_pipeline:.2f}")

    # # ==============================
    # # パッチ毎の統計量確認（ホーローの影響を調べるため）
    # # ※ config の設定に合わせて、patch_shape および stride_shape を使って走査
    # # ※ 今回は patch_shape = [128, 128, 128]、stride_shape = [128, 128, 128] と仮定
    # patch_shape = [128, 128, 128]
    # stride_shape = [128, 128, 128]
    # print("\n>> StandardPredictor 経由の結果（正規化前）のパッチ統計量")
    # # ※ もし複数パッチある場合は各パッチ毎の統計量を出力する
    # debug_patch_stats(prediction_predict, patch_shape, stride_shape, label="Predict.py Patch")
    
    # print("\n>> predict3dunet 経由の結果（正規化前）のパッチ統計量")
    # # ここでも HDF5 から読み込んだ生の予測値を対象とする
    # # ※ debug_patch_stats 内で patch 毎の統計量が表示される
    # # ※ 今回は全体が1パッチの場合も同じ結果になるはず
    # with h5py.File(prediction_file, 'r') as f:
    #     raw_pipeline_pred = np.squeeze(f['predictions'][:], axis=0)
    # debug_patch_stats(raw_pipeline_pred, patch_shape, stride_shape, label="predict3dunet Patch")

    # ==============================
    # 結果の比較
    # ==============================
    print("\n================== 結果のまとめ ==================")
    print(f"Raw  PSNR: {raw_psnr:.2f}")
    print(f"Validation 時の PSNR: {psnr_validation:.2f}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n[DEBUG] 総処理時間: {elapsed:.2f} 秒")


if __name__ == '__main__':
    main()

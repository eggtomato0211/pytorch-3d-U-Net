import os
import yaml
import subprocess
import re

# オリジナルの設定ファイル
original_config = r"C:\Users\Owner\mizusaki\pytorch-3dunet\resources\3DUnet_denoising\test_config_regression.yaml"

# `train-yaml` のベースディレクトリ
base_yaml_dir = r"C:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\predict-yaml"

# model_name とパラメータ設定
model_name = "patch=64_stride=16_fm=16_valpatch=64"
# predictするファイルのパス
test_file = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\val250\Number1.h5"

# YAML設定を読み込む
with open(original_config, "r") as f:
    config = yaml.safe_load(f)

# 設定変更
#f_mapの設定を書き換え, model_name によって変更
if "fm=16" in model_name:
    config["model"]["f_maps"] = [16, 32, 64, 128, 256]
elif "fm=32" in model_name:
    config["model"]["f_maps"] = [32, 64, 128, 256, 512]
elif "fm=64" in model_name:
    config["model"]["f_maps"] = [64, 128, 256, 512, 1024]

#patch_shapeの設定を書き換え, model_name によって変更, valpatchの値をそのまま使用
model_path = os.path.join(r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128", model_name, "best_checkpoint.pytorch")
config["model_path"] = model_path

#test_fileのHDFファイルのサイズをとってくる
import h5py
import numpy as np
with h5py.File(test_file, "r") as f:
    size = np.array(f["raw"]).shape

# サイズを確認
print(size)

# HDFファイルのサイズをpatch_shapeに設定
depth = size[0]
height = size[1]
width = size[2]

config["loaders"]["test"]["slice_builder"]["patch_shape"] = [depth, height, width] 
config["loaders"]["test"]["slice_builder"]["stride_shape"] = [depth, height, width]
config["loaders"]["test"]["file_paths"] = [test_file]

os.makedirs(base_yaml_dir, exist_ok=True)
yaml_path = os.path.join(base_yaml_dir, f"{model_name}.yaml")

# 新しい YAML 設定を保存
with open(yaml_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

# # `train3dunet` コマンドを実行
# subprocess.run(["poetry","run","predict3dunet", "--config", yaml_path])

prediction_file = test_file.replace(".h5", "_predictions.h5")

# 実行結果の確認(Label, Raw, Prediction)のPSNR比較
with h5py.File(test_file, 'r') as f:
    label_data = f['label'][:]
    raw_data = f['raw'][:]
with h5py.File(prediction_file, 'r') as f:
    prediction_data = f['predictions'][:]
    #label(128, 128, 128), raw(128, 128, 128), prediction(1, 128, 128, 128)なのでsqueeze()で次元を削減
    prediction_data = np.squeeze(prediction_data, axis=0)

#データを正規化
# ** 正規化の適用（バリデーションと同じスケール）**
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

label_data = normalize_data(label_data)
raw_data = normalize_data(raw_data)
prediction_data = normalize_data(prediction_data)

# Raw, PredicitonのPSNRの計算
from skimage.metrics import peak_signal_noise_ratio
data_range = label_data.max() - label_data.min()
raw_psnr = peak_signal_noise_ratio(label_data, raw_data, data_range=data_range)
prediction_psnr = peak_signal_noise_ratio(label_data, prediction_data, data_range=data_range)

print(f"Raw PSNR: {raw_psnr:.2f}")
print(f"Prediction PSNR: {prediction_psnr:.2f}")


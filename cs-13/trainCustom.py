import os
import yaml
import subprocess

# オリジナルの設定ファイル
original_config = r"C:\Users\Owner\mizusaki\pytorch-3dunet\resources\3DUnet_denoising\train_config_regression.yaml"

# `checkpoint_dir` のベースディレクトリ
base_checkpoint_dir = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128"
# `train-yaml` のベースディレクトリ
base_yaml_dir = r"C:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\train-yaml"

# パラメータ設定
patch_shape = [64, 64, 64]   # train 用の patch サイズ
stride_shape = [48, 48, 48]     # train の stride
val_patch_shape = [128, 128, 128]  # validation 用の patch サイズ
f_maps =  [16, 32, 64, 128, 256]   # f_maps の設定 [16, 32, 64, 128, 256] [32, 64, 128, 256, 512] [64, 128, 256, 512, 1024]

# `checkpoint_dir` と `yaml` の動的な名前
config_name = f"patch={patch_shape[0]}_stride={stride_shape[0]}_fm={f_maps[0]}_valpatch={val_patch_shape[0]}"
checkpoint_dir = os.path.join(base_checkpoint_dir, config_name)
yaml_path = os.path.join(base_yaml_dir, f"{config_name}.yaml")

# YAML設定を読み込む
with open(original_config, "r") as f:
    config = yaml.safe_load(f)

# 設定変更
config["trainer"]["checkpoint_dir"] = checkpoint_dir
config["model"]["f_maps"] = f_maps
config["loaders"]["train"]["slice_builder"]["patch_shape"] = patch_shape
config["loaders"]["train"]["slice_builder"]["stride_shape"] = stride_shape
config["loaders"]["val"]["slice_builder"]["patch_shape"] = val_patch_shape  # validation は val_patch_shape を使用
config["loaders"]["val"]["slice_builder"]["stride_shape"] = val_patch_shape  # validation の stride は patch と同じ

# `checkpoint_dir` と `train-yaml` フォルダを作成
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(base_yaml_dir, exist_ok=True)

# 新しい YAML 設定を保存
with open(yaml_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

# `train3dunet` コマンドを実行
subprocess.run(["poetry","run","train3dunet", "--config", yaml_path])

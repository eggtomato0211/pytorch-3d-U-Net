import subprocess
import os
import shutil
import re
import yaml

# 変更したいYAMLファイルのパス
yaml_path = r"C:\Users\Owner\mizusaki\pytorch-3dunet\resources\3DUnet_denoising\test_config_regression.yaml"

# 動的に変更したいテストデータのディレクトリとファイル名のフォーマット
base_path = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\test250"
file_prefix = "Number"
file_extension = ".h5"

# YAMLファイルを読み込み
try:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    original_model_path = config.get("model_path")
    if not original_model_path:
        raise ValueError("model_path is not defined in the YAML file.")
except Exception as e:
    print(f"Error reading YAML file: {e}")
    exit(1)

# 正規表現で 'stride=..._start=...' を抽出
match = re.search(r'stride=\d+_start=\d+', original_model_path)
if match:
    original_stride_part = match.group()
    print(f"Original stride part: {original_stride_part}")
else:
    raise ValueError("No stride part found in the model_path.")

# 移動先のベースディレクトリ
destination_dir_base = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\predictions"

# ファイル番号を自動生成 (1から250まで)
test_datasets = [os.path.join(base_path, f"{file_prefix}{i}{file_extension}") for i in range(1, 251)]

# 複数の `stride=..._start=...` 設定を試す
stride_list = [
    "stride=1_start=64",
    "stride=2_start=64", 
    "stride=4_start=64",
    "stride=8_start=64",
    "stride=16_start=64",
    "stride=32_start=64",
]  # ここに追加

for stride_part in stride_list:
    # base_path にあるpredict3dunetの実行結果を削除
    for file_name in os.listdir(base_path):
        if file_name.endswith("_predictions.h5"):
            file_path = os.path.join(base_path, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_name}: {e}")

    # `model_path` を変更
    new_model_path = re.sub(r'stride=\d+_start=\d+', stride_part, original_model_path)
    config["model_path"] = new_model_path

    # YAML を更新
    try:
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Updated model_path: {new_model_path}")
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        continue

    # `destination_dir_base/stride_part` の形式に変更
    destination_dir = os.path.join(destination_dir_base, stride_part)

    # テストデータごとに処理
    for i, test_file in enumerate(test_datasets, start=1):
        # `test.file_paths` を変更
        config["loaders"]["test"]["file_paths"] = [test_file]

        # YAML を更新
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error writing YAML file: {e}")
            continue

        # `predict3dunet` を実行
        cmd = ["poetry", "run", "predict3dunet", "--config", yaml_path]
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing predict3dunet for {test_file}: {e.stderr}")
            continue

        # 進捗表示
        if i % 10 == 0:
            print(f"{i} files processed for {stride_part}...")

    # 移動先ディレクトリを作成
    os.makedirs(destination_dir, exist_ok=True)

    # `source_dir` 内の `_predictions.h5` を `destination_dir` に移動
    source_dir = base_path
    for file_name in os.listdir(source_dir):
        if file_name.endswith("_predictions.h5"):
            source_path = os.path.join(source_dir, file_name)
            destination_path = os.path.join(destination_dir, file_name)
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved {source_path} to {destination_path}")
            except Exception as e:
                print(f"Error moving file {file_name}: {e}")

    print(f"All files have been moved for {stride_part}.")

print("All predictions completed for all stride configurations.")

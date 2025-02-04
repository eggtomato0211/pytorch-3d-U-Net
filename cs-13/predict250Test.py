import subprocess
import os
import shutil
import re
import yaml
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 事前準備 ---
mode = "val"  # "train" or "test"
yaml_path = r"C:\Users\Owner\mizusaki\pytorch-3dunet\resources\3DUnet_denoising\test_config_regression.yaml"
base_path = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\{mode}250"
destination_dir_base = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\predictions_{mode}"
checkpoint_base_path = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128"

file_prefix = "Number"
file_extension = ".h5"

# YAML を読み込む
try:
    with open(yaml_path, "r") as f:
        base_config = yaml.safe_load(f)
except Exception as e:
    print(f"YAMLファイルの読み込みエラー: {e}")
    exit(1)

# テストデータ一覧（例：Number1.h5 ~ Number250.h5）
test_datasets = [os.path.join(base_path, f"{file_prefix}{i}{file_extension}") for i in range(1, 251)]

# 使用する model_name 設定
model_names = [
    "patch=64_stride=16_fm=16_valpatch=64",
]

# 予測実行用の関数
def run_prediction(test_file, config, model_name, gpu_id, tmp_yaml_path):
    """
    ・config のディープコピーを作成し、 test_file と model_path を更新  
    ・一時的な YAML ファイルを書き出す  
    ・環境変数 CUDA_VISIBLE_DEVICES で gpu_id を指定して predict3dunet を実行する  
    ・実行後、一時 YAML を削除
    """
    local_config = copy.deepcopy(config)

    # `train3dunet` で作成された `best_checkpoint.pytorch` を特定
    checkpoint_dir = os.path.join(checkpoint_base_path, model_name)
    model_path = os.path.join(checkpoint_dir, "best_checkpoint.pytorch")

    # YAML の設定を書き換え
    local_config["loaders"]["test"]["file_paths"] = [test_file]
    local_config["model_path"] = model_path

    #f_mapの設定を書き換え, model_name によって変更
    if "fm=16" in model_name:
        local_config["model"]["f_maps"] = [16, 32, 64, 128, 256]
    elif "fm=32" in model_name:
        local_config["model"]["f_maps"] = [32, 64, 128, 256, 512]
    elif "fm=64" in model_name:
        local_config["model"]["f_maps"] = [64, 128, 256, 512, 1024]
    
    #patch_shapeの設定を書き換え, model_name によって変更, valpatchの値をそのまま使用
    valpatch_shape = int(re.search(r"valpatch=(\d+)", model_name).group(1))
    local_config["loaders"]["test"]["slice_builder"]["patch_shape"] = [valpatch_shape, valpatch_shape, valpatch_shape] 
    local_config["loaders"]["test"]["slice_builder"]["stride_shape"] = [valpatch_shape, valpatch_shape, valpatch_shape]

    # 一時的な YAML ファイルとして保存
    with open(tmp_yaml_path, "w") as f:
        yaml.dump(local_config, f, default_flow_style=False)

    # 実行コマンドと環境変数の設定
    cmd = ["poetry", "run", "predict3dunet", "--config", tmp_yaml_path]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        print(f"GPU{gpu_id} で {test_file} を処理中...")
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error for {test_file}: {e.stderr}"
    finally:
        # 一時ファイルの削除（必要ならコメントアウトしてください）
        if os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)
    return output

# --- 並列実行部分 ---
# 各 model_name 設定ごとに処理
for model_name in model_names:
    print(f"処理中の model_name 設定: {model_name}")

    # 事前に古い予測結果 (_predictions.h5) を削除
    for file_name in os.listdir(base_path):
        if file_name.endswith("_predictions.h5"):
            try:
                os.remove(os.path.join(base_path, file_name))
                print(f"Deleted {file_name}")
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")

    # 移動先ディレクトリ（model_nameごと）を作成
    destination_dir = os.path.join(destination_dir_base, model_name)
    os.makedirs(destination_dir, exist_ok=True)

    # 並列実行（max_workers=4 で GPU0～GPU3 を使用）
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, test_file in enumerate(test_datasets, start=1):
            # GPU を (i-1)%4 によって順番に割り当て
            gpu_id = (i - 1) % 4
            # 一時的な YAML ファイル名（model_name とテスト番号で一意にする）
            tmp_yaml = os.path.join(base_path, f"tmp_config_{model_name}_{i}.yaml")
            futures.append(executor.submit(run_prediction, test_file, base_config, model_name, gpu_id, tmp_yaml))
            
            # 進捗表示
            if i % 10 == 0:
                print(f"{i} ファイル処理中...")

        # 各タスクの完了を待って結果を表示
        for future in as_completed(futures):
            print(future.result())

    # 並列処理完了後、生成された _predictions.h5 ファイルを移動
    for file_name in os.listdir(base_path):
        if file_name.endswith("_predictions.h5"):
            source_path = os.path.join(base_path, file_name)
            destination_path = os.path.join(destination_dir, file_name)
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved {source_path} to {destination_path}")
            except Exception as e:
                print(f"Error moving file {file_name}: {e}")

    print(f"model_name 設定 {model_name} の全予測処理完了。\n")

print("全ての model_name 設定に対する予測処理が完了しました。")

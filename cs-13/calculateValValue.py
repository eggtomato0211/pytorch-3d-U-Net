import os
import yaml
import h5py
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

# pytorch-3dunet 内の各種モジュール（必要に応じてパス等を調整してください）
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import load_checkpoint
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.augment.transforms import Normalize

# ======================================
# 設定パスの指定
# ======================================
val_dir = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\val250"
model_checkpoint = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128\patch=64_stride=48_fm=16_valpatch=128\best_checkpoint.pytorch"
original_config = r"C:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\train-yaml\patch=64_stride=48_fm=16_valpatch=128.yaml"

# ======================================
# YAML 設定の読み込みとモデルのセットアップ
# ======================================
with open(original_config, "r") as f:
    config = yaml.safe_load(f)

# デバイスの設定（GPU があれば GPU を使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル生成しチェックポイントをロード
model = get_model(config['model'])
checkpoint = load_checkpoint(model_checkpoint, model)
model = model.to(device)
model.eval()

# 必要に応じて loss, evaluation criterion を取得（ここでは PSNR を算出するため、skimage の関数を利用）
loss_criterion = get_loss_criterion(config)
eval_criterion = get_evaluation_metric(config)  # ※設定によっては PSNR などの関数が返る場合もあります

# ======================================
# Normalize インスタンスの作成
# ======================================
normalize_transform = Normalize()

# ======================================
# HDF5 データのロード関数
# ======================================
def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"📂 ファイル情報: {list(f.keys())}")
        # 各ファイル内のキーは 'raw' と 'label' と仮定
        label_data = f['label'][:]
        raw_data = f['raw'][:]
    
    # 入力データにも label にも正規化（学習時と同じ前処理を適用）
    raw_data = normalize_transform(raw_data)
    label_data = normalize_transform(label_data)
    return label_data, raw_data

# ======================================
# validation 用関数（trainer.validate() の流れを模して実装）
# ======================================
def validate_model(model, file_list, device):
    # 評価スコアを蓄積するリスト
    val_scores = []
    # ここでは loss の計算も可能ですが、例では PSNR のみを計算しています
    
    # 評価モードで no_grad() で実行
    model.eval()
    with torch.no_grad():
        for i, file_path in enumerate(file_list):
            print(f"🔍 Validation iteration {i}: {os.path.basename(file_path)}")
            
            # HDF5 ファイルからデータを取得
            label_data, raw_data = load_hdf5_data(file_path)
            
            # 入力テンソルへの変換
            # ※ここでは raw_data が 3次元 (D, H, W) の場合、チャンネル次元とバッチ次元を追加して (1, 1, D, H, W) とする例です
            input_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # ---------------------------
            # 推論（forward pass）
            # ---------------------------
            output_tensor = model(input_tensor)
            
            # ※必要に応じて loss の計算も可能
            # loss = loss_criterion(output_tensor, target_tensor)
            
            # 出力テンソルを numpy 配列に変換（バッチ次元、チャンネル次元を除去）
            output = output_tensor.squeeze(0).squeeze(0).cpu().numpy()

            print(f"Output min/max: {output.min()} / {output.max()}")
            
            # ※場合によっては、出力に後処理（例：再度正規化）を施す
            output = normalize_transform(output)

            print(f"Output min/max: {output.min()} / {output.max()}")
            
            # ---------------------------
            # PSNR の計算
            # ---------------------------
            psnr_value = peak_signal_noise_ratio(
                label_data, output,
                data_range=np.max(label_data) - np.min(label_data)
            )
            print(f"📊 PSNR: {psnr_value}")
            val_scores.append(psnr_value)
            
            # 100イテレーション毎に画像などのログ出力を行う（必要に応じて実装）
            if i % 100 == 0:
                # 例: ログ用の画像を保存・TensorBoard に出力など
                pass
            
            # ※必要であれば、validation ループの途中で break する条件も追加可能
            # if some_condition:
            #     break
    
    # 蓄積した各バッチの PSNR の平均値を返す
    avg_score = np.mean(val_scores)
    return avg_score

# ======================================
# validation ファイル一覧の取得(_predictions.h5 は排除)
# ======================================
h5_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.h5') and not f.endswith('_predictions.h5')]

print(f"📂 Validation ファイル数: {len(h5_files)}"
      f"\n📁 Validation ファイルリスト: {h5_files}")

# ======================================
# validation の実行
# ======================================
avg_psnr = validate_model(model, h5_files, device)

print("\n==============================")
print(f"📊 `best_checkpoint.pytorch` のバリデーション時の 平均 PSNR: {avg_psnr}")
print(f"📌 `best_eval_score`: {checkpoint.get('best_eval_score', '未定義')}")
print("==============================")

# ======================================
# 一致チェック
# ======================================
if 'best_eval_score' in checkpoint and np.isclose(avg_psnr, checkpoint['best_eval_score'], atol=1e-5):
    print("✅ `best_checkpoint.pytorch` のバリデーションスコアは `best_eval_score` と完全に一致しました！")
else:
    print("❌ `best_checkpoint.pytorch` のバリデーションスコアは `best_eval_score` と一致しません！ 設定を確認してください。")

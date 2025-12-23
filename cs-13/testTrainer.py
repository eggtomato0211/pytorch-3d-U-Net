import os
import yaml
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# pytorch-3dunet 内の trainer モジュールから create_trainer をインポート
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.model import get_model
import h5py

def load_model(checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = get_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def compute_psnr(gt, pred):
    data_range = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)

def main():
    # === 設定ファイルとチェックポイントのパスを指定 ===
    config_path = r'C:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\train-yaml\patch=64_stride=48_fm=16_valpatch=128.yaml'
    best_checkpoint_path = r'C:\Users\Owner\mizusaki\pytorch-3dunet\cs-13\train-yaml\patch=64_stride=48_fm=16_valpatch=128.yaml'
    
    # YAML 設定ファイルを読み込む
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # trainer セクションに resume キーを設定することで、best_checkpoint を読み込む
    config['trainer']['resume'] = best_checkpoint_path

    # UNetTrainer のインスタンスを作成
    trainer = create_trainer(config)
    
    # ----- バリデーションの実行 -----
    # 改変した validate() は (PSNR, label_np, input_np) を返す
    val_score, label_np, input_np = trainer.validate()
    print(f"Validation Score (PSNR) with best_checkpoint: {val_score}")

    # --- モデルの読み込み ---
    print(">> モデルの読み込み")
    model = load_model(best_checkpoint_path, config)

    # --- PSNR: Raw と Label の比較 ---
    # （ここでは、validation 用データの正解と入力を直接比較して PSNR を計算）
    raw_psnr = compute_psnr(label_np, input_np)
    print(f"[DEBUG] 【Raw と Label の PSNR】: {raw_psnr:.2f}")

    # ==============================
    # ① 直接推論（Validation 時の1バッチに対して、validation 用データを入力して推論）
    # ==============================
    # 例えば、取得した input_np をモデルに投入して直接推論を行う
    # ※入力の形状がモデルの期待する形状（バッチ次元やチャンネル次元）と一致するか注意してください
    input_tensor = torch.tensor(input_np, dtype=torch.float32)

    # 必要に応じて、バッチ次元・チャンネル次元の追加（例: (B,C,H,W) もしくは (B,1,H,W)）
    # 以下は例です。実際のデータ形状に合わせて調整してください。
    if input_tensor.ndim == 3:  # (D, H, W) の場合
        input_tensor = input_tensor[None, None, ...]  # → (1,1,D,H,W)
    elif input_tensor.ndim == 4:  # (C, D, H, W) の場合
        input_tensor = input_tensor[None, ...]  # → (1,C,D,H,W)
    
    # モデルが GPU 対応の場合、必要なら転送
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 出力テンソルを NumPy 配列へ変換（バッチ・チャンネルの次元を除去するなど、適宜調整）
    output_np = output_tensor.cpu().numpy()
    # ここでは、出力と正解 label_np の形状が一致している前提で PSNR を計算
    direct_psnr = compute_psnr(label_np, output_np)
    print(f"Direct inference PSNR: {direct_psnr:.2f}")

    # val_score と direct_psnr が一致しているか確認、あっている場合は明快にわかりやすく表示する
    if val_score == direct_psnr:
        print(f"🎉 Validation Score と Direct Inference の PSNR が一致しました！ValScore{val_score}, DirectInference{direct_psnr}")
    else:
        print(f"😢 Validation Score と Direct Inference の PSNR が一致しませんでした...ValScore{val_score}, DirectInference{direct_psnr}")

    # ==============================

    # ==============================
    # ① 直接推論（hdfファイルを使って推論）
    # ==============================
    # 例えば、取得した input_np をモデルに投入して直接推論を行う
    def load_hdf5_data(file_path):
        with h5py.File(file_path, 'r') as f:
            label_data = f['label'][:]
            raw_data   = f['raw'][:]
        return label_data, raw_data
    
    def normalize_data(data):
        """最小値と最大値を使って正規化し、-1～1の範囲に収める"""
        min_value = data.min()
        max_value = data.max()
        return (data - min_value) / (max_value - min_value) * 2 - 1
        
    #configファイルの設定を読み込む
    hdf5_file_folder = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\valSimple"
    #フォルダ内のhdf5ファイルを読み込む(endwithでhdf5ファイルを指定)
    h5_files = [os.path.join(hdf5_file_folder, f) for f in os.listdir(hdf5_file_folder) if f.endswith('.h5') and not f.endswith('_predictions.h5')]

    # 1つしかないので、0番目のファイルを読み込む
    label_data, raw_data = load_hdf5_data(h5_files[0])

    def debug_stats(data, name):
        """データの shape、min、max、mean を出力"""
        print(f"[DEBUG] {name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    
    debug_stats(label_data, 'label_data')
    debug_stats(raw_data, 'raw_data')

    # --- 正規化 ---
    label_data = normalize_data(label_data)
    raw_data   = normalize_data(raw_data)

    # label_npとlabel_dataの比較
    print(f"label_np: {label_np.shape}, label_data: {label_data.shape}")
    # label_np: (1, 1, 128, 128, 128), label_data: (128, 128, 128)なので、label_dataに次元を合わせるために、label_npの次元を変更
    label_np = np.squeeze(label_np)

    # データが一致しているか確認
    print(f"label_np: {label_np.shape}, label_data: {label_data.shape}")
    # 3次元配列の値がすべて一致しているか確認
    print(f"Label data and label_np are equal: {np.allclose(label_data, label_np)}")

    # raw_dataとinput_npの比較
    print(f"input_np: {input_np.shape}, raw_data: {raw_data.shape}")
    # input_np: (1, 1, 128, 128, 128), raw_data: (128, 128, 128)なので、raw_dataに次元を合わせるために、input_npの次元を変更
    input_np = np.squeeze(input_np)

    # データが一致しているか確認
    print(f"input_np: {input_np.shape}, raw_data: {raw_data.shape}")
    # 3次元配列の値がすべて一致しているか確認
    print(f"Input data and input_np are equal: {np.allclose(raw_data, input_np)}")

    # 一致している場合、モデルの予測を行い、PSNRを計算
    if np.allclose(label_data, label_np):
        # raw_data を入力を確認
        print(f"[DEBUG] Raw Data: shape={raw_data.shape}, min={raw_data.min():.4f}, max={raw_data.max():.4f}, mean={raw_data.mean():.4f}")

        # 入力データの次元を変更
        input_tensor = torch.tensor(raw_data, dtype=torch.float32)

        if input_tensor.ndim == 3:
            input_tensor = input_tensor[None, None, ...]
        elif input_tensor.ndim == 4:
            input_tensor = input_tensor[None, ...]
        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # input_tensorの内容を確認
        debug_stats(input_tensor.cpu().numpy(), "Validation Input Tensor")
        # output_tensorの内容を確認
        debug_stats(output_tensor.cpu().numpy(), "Validation Prediction Tensor")

        output_np = output_tensor.cpu().numpy()
        # output_npの次元を変更
        output_np = np.squeeze(output_np)
        inference_psnr = compute_psnr(label_data, output_np)
        print(f"Model inference PSNR: {inference_psnr:.2f}")
    
    # val_score と inference_psnr が一致しているか確認、あっている場合は明快にわかりやすく表示する
    if val_score == inference_psnr:
        print(f"🎉 Validation Score と Direct Inference の PSNR が一致しました！ValScore{val_score}, Model Inference{inference_psnr}")
    else:
        print(f"😢 Validation Score と Direct Inference の PSNR が一致しませんでした...ValScore{val_score}, Model Inference{inference_psnr}")

if __name__ == '__main__':
    main()

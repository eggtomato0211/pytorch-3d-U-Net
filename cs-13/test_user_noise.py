import os
import h5py
from generate_noisy_data import process

# テスト用の設定
DATA_DIR = r"D:/nosaka/data/3d-holography_output/Train"
OUTPUT_DIR = r"D:/nosaka/data/3d-holography_output/Train_user_noisy_test"
TARGET_PSNR = 18.0
TEST_FILE = "128images_10plots_fixed_randomFalse_NumberFrom1025.h5"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1ファイルだけ処理するように一時的なディレクトリを作成するか、
    # 直接プロシージャを呼ぶ
    print(f"🔄 ユーザーの新規ノイズモデルでテスト実行中: {TEST_FILE}")
    
    in_path = os.path.join(DATA_DIR, TEST_FILE)
    with h5py.File(in_path, "r") as f:
        raw = f["raw"][:]
        label = f["label"][:]
        
    from generate_noisy_data import tune_noise
    noisy = tune_noise(raw, target_psnr=TARGET_PSNR)
    
    out_path = os.path.join(OUTPUT_DIR, TEST_FILE.replace(".h5", "_user_noisy.h5"))
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=noisy)
        f.create_dataset("label", data=label)
        
    print(f"✅ 完了: {out_path}")

if __name__ == "__main__":
    main()

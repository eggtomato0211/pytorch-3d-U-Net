import os
import h5py
import numpy as np
import scipy.ndimage as ndimage
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import random

# ==========================================
# ⚙️ 設定エリア (環境に合わせて書き換えてください)
# ==========================================
# ベースとなる入力ディレクトリ (この中に Train, Val, Test がある想定)
BASE_INPUT_DIR = Path(r"D:/nosaka/data/3d-holography_output")

# 出力先のベースディレクトリ
BASE_OUTPUT_DIR = Path(r"D:/nosaka/data/3d-holography_output")

# 実行時刻 (フォルダ名用)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# 処理する対象とモードの設定
# "mode": "random" -> 学習用 (パラメータを散らす)
# "mode": "target" -> 評価用 (マウス脳に近い 0.15 固定)
DATASETS_TO_PROCESS = [
    {"name": "Train", "mode": "random"},  # 学習用: ランダム
    {"name": "Val",   "mode": "target"},  # 検証用: マウス脳設定固定
    {"name": "Test",  "mode": "target"},  # テスト用: マウス脳設定固定
    # {"name": "Test", "mode": "hard"},  # (任意) テスト用: 激ムズ設定を作りたい場合
]

# --- 共通パラメータ (固定) ---
OBJECT_PROTECTION_THRESHOLD = 0.05  # 信号保護マスクの閾値
HAZE_CLOUD_SIGMA = 50.0             # 背景ムラの周波数
# ==========================================


def get_params_by_mode(mode):
    """
    モードに応じて、霧の濃さとノイズ量を決定する
    """
    if mode == "random":
        # 【学習用】 汎用性を高めるため、広くランダムに振る
        # 霧: 0.05 (薄) ~ 0.35 (濃)
        # ノイズ: 200 (汚) ~ 1000 (綺)
        intensity = random.uniform(0.05, 0.35)
        noise_peak = random.randint(200, 1000)
        glow_sigma = random.uniform(20.0, 40.0)
        return intensity, noise_peak, glow_sigma

    elif mode == "target":
        # 【評価用】 マウス脳の実データに見た目を寄せる
        intensity = 0.15  # 固定
        noise_peak = 600  # 固定
        glow_sigma = 30.0 # 固定
        return intensity, noise_peak, glow_sigma
    
    elif mode == "hard":
        # 【ストレステスト用】 非常に厳しい条件
        intensity = 0.30
        noise_peak = 300
        glow_sigma = 30.0
        return intensity, noise_peak, glow_sigma
    
    else:
        # デフォルト (Targetと同じ)
        return 0.15, 600, 30.0


def add_mist_effect(x, intensity, noise_peak, glow_sigma):
    """
    指定されたパラメータで霧とノイズを付加するコア関数
    """
    x = x.astype(np.float32)
    vmin, vmax = x.min(), x.max()
    if vmax == vmin: return x

    # 0-1正規化
    x_norm = (x - vmin) / (vmax - vmin)

    # 1. 霧 (Haze) の生成
    # 信号由来の散乱 (Glow)
    glow = ndimage.gaussian_filter(x_norm, sigma=glow_sigma)
    
    # 背景のムラ (Cloud)
    shape = x.shape
    noise_low = ndimage.gaussian_filter(np.random.rand(*shape).astype(np.float32), sigma=HAZE_CLOUD_SIGMA)
    noise_mid = ndimage.gaussian_filter(np.random.rand(*shape).astype(np.float32), sigma=HAZE_CLOUD_SIGMA / 2.0)
    cloud = (noise_low + 0.5 * noise_mid)
    
    # Cloudの正規化
    c_min, c_max = cloud.min(), cloud.max()
    if c_max > c_min: cloud = (cloud - c_min) / (c_max - c_min)

    # 散乱場の合成
    atmosphere = (glow * 0.7) + (cloud * 0.3)
    
    # マスク処理 (信号保護)
    protection_mask = 1.0 - np.clip(x_norm / OBJECT_PROTECTION_THRESHOLD, 0.0, 1.0)
    
    # 霧を合成
    mist = atmosphere * protection_mask * intensity
    out_norm = x_norm + mist
    out_norm = np.clip(out_norm, 0.0, 1.0)

    # 2. センサーノイズ (Shot Noise)
    if noise_peak > 0:
        lam = np.clip(out_norm * float(noise_peak), 0.0, None)
        out_norm = np.random.poisson(lam).astype(np.float32) / float(noise_peak)
    
    out_norm = np.clip(out_norm, 0.0, 1.0)

    # 元のスケールに戻す
    return out_norm * (vmax - vmin) + vmin


def process_directory(config):
    """
    1つのディレクトリ(Train/Val/Test)を処理する関数
    """
    dir_name = config["name"]
    mode = config["mode"]
    
    input_dir = BASE_INPUT_DIR / dir_name
    
    # 出力フォルダ名: 例 "Train_misty_random_20251226_1430"
    suffix = f"_{mode}_{TIMESTAMP}" if mode != "target" else f"_{TIMESTAMP}"
    output_dir_name = f"{dir_name}_misty{suffix}"
    output_dir = BASE_OUTPUT_DIR / output_dir_name
    
    if not input_dir.exists():
        print(f"⚠️ スキップ: 入力フォルダが見つかりません -> {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルリスト取得
    files = sorted([f for f in input_dir.glob("*.h5") if "predict" not in f.name])
    
    print(f"\n🚀 Processing: {dir_name} -> {output_dir_name}")
    print(f"   Mode: {mode} (Files: {len(files)})")
    
    for file_path in tqdm(files, desc=f"   Building {dir_name}"):
        try:
            with h5py.File(file_path, "r") as f:
                if "raw" not in f or "label" not in f: continue
                raw = f["raw"][:]
                label = f["label"][:]
            
            # ★ ここでモードに応じたパラメータを取得
            intensity, noise_peak, glow_sigma = get_params_by_mode(mode)
            
            # ノイズ付加
            noisy_raw = add_mist_effect(raw, intensity, noise_peak, glow_sigma)
            
            # 保存
            out_name = file_path.stem + "_misty.h5"
            out_path = output_dir / out_name
            
            with h5py.File(out_path, "w") as f:
                f.create_dataset("raw", data=noisy_raw)
                f.create_dataset("label", data=label)
                
                # 記録用属性 (後で確認できるようにパラメータを保存しておくと便利)
                f["raw"].attrs["haze_intensity"] = intensity
                f["raw"].attrs["sensor_noise_peak"] = noise_peak
        
        except Exception as e:
            print(f"❌ Error in {file_path.name}: {e}")

    print(f"✅ 完了: {output_dir}")


def main():
    print(f"=== 3D Holography Misty Dataset Generator ===")
    print(f"🕒 Timestamp: {TIMESTAMP}")
    print(f"📂 Base Input: {BASE_INPUT_DIR}")
    print("="*50)
    
    # 設定された各データセットを順番に処理
    for config in DATASETS_TO_PROCESS:
        process_directory(config)
        
    print("\n🎉 全ての処理が完了しました！ お疲れ様でした。")

if __name__ == "__main__":
    main()
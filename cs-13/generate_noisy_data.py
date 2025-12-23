import os
import h5py
import numpy as np
import scipy.ndimage as ndimage
from skimage.metrics import peak_signal_noise_ratio

# ==========================================
# ⚙️ 最終決定版パラメータ (ここをいじるだけでOK)
# ==========================================
DEFAULT_DATA_DIR = r"D:/nosaka/data/3d-holography_output/Test"
DEFAULT_OUTPUT_DIR = r"D:/nosaka/data/3d-holography_output/Test_misty"

# 1. 霧の濃さ (Intensity)
# 0.1 ~ 0.2 が「くっきり + モヤ」の黄金ラインです。
# 0.05: ほぼ見えない / 0.15: 良い感じ / 0.3: 濃すぎる
HAZE_INTENSITY = 0.2

# 2. データの保護 (Protection)
# これがある限り、棒状のデータ(明るい部分)は汚れません。
# 0.05 (=5%) 以上の輝度がある場所は保護されます。
OBJECT_PROTECTION_THRESHOLD = 0.05

# 3. 霧の広がり (Glow)
# データの周りにどれくらい漂わせるか。
HAZE_GLOW_SIGMA = 30.0

# 4. 背景のムラ (Cloud)
HAZE_CLOUD_SIGMA = 50.0

# 5. センサーノイズ (隠し味)
# 霧だけだとCGっぽすぎるので、ごくわずかに砂嵐を混ぜてリアリティを出します。
SENSOR_NOISE_PEAK = 500  # 値が大きいほどノイズは減ります (S/N比なので)
# ==========================================


def add_mist_effect(x):
    """
    パラメータに従って、霧と少量のセンサーノイズを付加する関数
    (PSNRによる調整ループは廃止)
    """
    x = x.astype(np.float32)
    vmin, vmax = x.min(), x.max()
    if vmax == vmin: return x

    # 0-1正規化
    x_norm = (x - vmin) / (vmax - vmin)

    # --- 1. 霧 (Haze) の生成 ---
    glow = ndimage.gaussian_filter(x_norm, sigma=HAZE_GLOW_SIGMA)
    
    shape = x.shape
    noise_low = ndimage.gaussian_filter(np.random.rand(*shape).astype(np.float32), sigma=HAZE_CLOUD_SIGMA)
    noise_mid = ndimage.gaussian_filter(np.random.rand(*shape).astype(np.float32), sigma=HAZE_CLOUD_SIGMA / 2.0)
    cloud = (noise_low + 0.5 * noise_mid)
    
    c_min, c_max = cloud.min(), cloud.max()
    if c_max > c_min: cloud = (cloud - c_min) / (c_max - c_min)

    # 霧の成分
    atmosphere = (glow * 0.7) + (cloud * 0.3)
    
    # マスク処理 (データを保護)
    protection_mask = 1.0 - np.clip(x_norm / OBJECT_PROTECTION_THRESHOLD, 0.0, 1.0)
    
    # 霧を合成
    mist = atmosphere * protection_mask * HAZE_INTENSITY
    out_norm = x_norm + mist
    out_norm = np.clip(out_norm, 0.0, 1.0)

    # --- 2. 隠し味のセンサーノイズ (Shot Noise) ---
    # 霧の上からごく薄くかける
    if SENSOR_NOISE_PEAK > 0:
        lam = np.clip(out_norm * float(SENSOR_NOISE_PEAK), 0.0, None)
        out_norm = np.random.poisson(lam).astype(np.float32) / float(SENSOR_NOISE_PEAK)
    
    out_norm = np.clip(out_norm, 0.0, 1.0)

    # 元のスケールに戻す
    return out_norm * (vmax - vmin) + vmin


def process(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".h5") and "predict" not in f.lower()]
    print(f"📂 Mode: Direct Parameter Control (No PSNR Loop)")
    print(f"🌫️ Haze Intensity: {HAZE_INTENSITY}")
    print(f"🛡️ Protection: {OBJECT_PROTECTION_THRESHOLD}")
    print("-" * 50)

    for i, fname in enumerate(files):
        in_path = os.path.join(data_dir, fname)
        print(f"[{i+1}/{len(files)}] 🔄 {fname} ... ", end="")

        try:
            with h5py.File(in_path, "r") as f:
                raw = f["raw"][:]
                label = f["label"][:]

            # 一発変換
            noisy = add_mist_effect(raw)
            
            # PSNRは「結果確認」のためだけに計算（制御には使わない）
            data_range = raw.max() - raw.min()
            final_psnr = peak_signal_noise_ratio(raw, noisy, data_range=data_range)

            # 保存
            out_name = os.path.splitext(fname)[0] + "_misty.h5"
            out_path = os.path.join(output_dir, out_name)
            with h5py.File(out_path, "w") as f:
                f.create_dataset("raw", data=noisy)
                f.create_dataset("label", data=label)

            print(f"✅ (Result PSNR: {final_psnr:.2f}dB)")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    process(DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR)
import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio

def add_composite_noise(x, haze_intensity=0.2, haze_sigma=12, peak=80, read_std=0.01, dc_offset=0.05):
    """
    Composite noise model:
    1. Low-frequency haze (Gaussian filter)
    2. DC Offset (Background lift)
    3. Shot noise (Poisson)
    4. Read noise (Gaussian)
    """
    x = x.astype(np.float32)
    vmin, vmax = x.min(), x.max()
    if vmax == vmin:
        return x.copy()

    # Normalize to 0-1 range for internal calculations
    x01 = (x - vmin) / (vmax - vmin)
    
    # --- 1. Low-frequency "Cloud/Haze" Noise (Moyatto) ---
    # Mouse brain autofluorescence / TIE artifacts simulation
    noise_low = np.random.normal(0, 1, size=x.shape).astype(np.float32)
    haze = gaussian_filter(noise_low, sigma=haze_sigma)
    # Normalize haze to [0, 1]
    haze = (haze - haze.min()) / (haze.max() - haze.min())
    x_hazy = x01 + (haze * haze_intensity)
    
    # --- 2. DC Offset (Background lift) ---
    # Stray light / Baseline simulation
    x_base = x_hazy + dc_offset
    
    # --- 3. Shot Noise (Poisson) & Read Noise (Gaussian) (Zarazara) ---
    # Pulse noise depending on intensity, plus sensor read noise
    lam = np.clip(x_base * float(peak), 0.0, None)
    noisy = np.random.poisson(lam).astype(np.float32) / float(peak)
    noisy += np.random.normal(0.0, read_std, size=x.shape).astype(np.float32)
    
    # Clip and restore original scale
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy * (vmax - vmin) + vmin

def compute_psnr(gt, noisy):
    data_range = float(gt.max() - gt.min()) or 1.0
    return peak_signal_noise_ratio(gt, noisy, data_range=data_range)

# Input/Output settings
DATA_DIR = r"D:/nosaka/data/3d-holography_output/Train"
OUTPUT_DIR = r"D:/nosaka/data/3d-holography_output/Train_composite_test"
FILENAME = "128images_10plots_fixed_randomFalse_NumberFrom1025.h5"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    in_path = os.path.join(DATA_DIR, FILENAME)
    out_path = os.path.join(OUTPUT_DIR, FILENAME.replace(".h5", "_composite.h5"))

    print(f"🔄 処理開始 (複合ノイズモデル): {FILENAME}")

    with h5py.File(in_path, "r") as f:
        raw = f["raw"][:]
        label = f["label"][:]

    # Generate composite noisy data
    noisy = add_composite_noise(raw)
    
    psnr = compute_psnr(raw, noisy)
    print(f"✅ 複合ノイズ 追加完了 -> PSNR={psnr:.2f} dB")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=noisy)
        f.create_dataset("label", data=label)

    print(f"💾 保存完了: {out_path}")

if __name__ == "__main__":
    main()

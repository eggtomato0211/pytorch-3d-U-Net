import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio

def add_hazy_fog(x, fog_intensity=0.3, sigma=10):
    """
    Add haze/fog effect using low-frequency noise (Gaussian blurred noise)
    """
    x = x.astype(np.float32)
    vmin, vmax = x.min(), x.max()
    if vmax == vmin:
        return x.copy()

    # Normalize to 0-1
    x01 = (x - vmin) / (vmax - vmin)
    
    # Create 3D smoke/fog layer
    # 3D Gaussian blurred noise creates "blobs" of light/haze
    noise = np.random.normal(0, 1, size=x.shape).astype(np.float32)
    fog = gaussian_filter(noise, sigma=sigma)
    
    # Normalize fog to [0, 1]
    fog = (fog - fog.min()) / (fog.max() - fog.min())
    
    # Add fog to the signal (additive haze)
    # This will create "hazy/cloudy" bright regions
    noisy = x01 + (fog * fog_intensity)
    
    # Clip and restore range
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy * (vmax - vmin) + vmin

def compute_psnr(gt, noisy):
    data_range = float(gt.max() - gt.min()) or 1.0
    return peak_signal_noise_ratio(gt, noisy, data_range=data_range)

# Use one specific file for the test
DATA_DIR = r"D:/nosaka/data/3d-holography_output/Train"
OUTPUT_DIR = r"D:/nosaka/data/3d-holography_output/Train_hazy_test"
FILENAME = "128images_10plots_fixed_randomFalse_NumberFrom1025.h5"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    in_path = os.path.join(DATA_DIR, FILENAME)
    out_path = os.path.join(OUTPUT_DIR, FILENAME.replace(".h5", "_hazy.h5"))

    print(f"🔄 処理開始 (Hazy効果): {FILENAME}")

    with h5py.File(in_path, "r") as f:
        raw = f["raw"][:]
        label = f["label"][:]

    # Experiment with fog intensity and blur sigma
    # intensity=0.4 and sigma=8-12 usually gives nice cloud blobs
    noisy = add_hazy_fog(raw, fog_intensity=0.4, sigma=12)
    
    psnr = compute_psnr(raw, noisy)
    print(f"✅ Hazy効果 追加完了 -> PSNR={psnr:.2f} dB")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=noisy)
        f.create_dataset("label", data=label)

    print(f"💾 保存完了: {out_path}")

if __name__ == "__main__":
    main()

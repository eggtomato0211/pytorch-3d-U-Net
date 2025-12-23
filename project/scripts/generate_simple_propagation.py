import numpy as np
import scipy.ndimage
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import h5py
import os
import argparse
from tqdm import tqdm
import random

class PaperSimulationGenerator:
    def __init__(self):
        # 論文 3.2.1 学習データセットの記述に基づくパラメータ
        self.sim_shape = (128, 32, 32)  # (Z, Y, X) 計算時のサイズ 
        self.final_shape = (128, 128, 128) # 最終的なサイズ 
        
        # 光学パラメータ（論文や一般的実験条件からの推定）
        self.pixel_size = 4.0   # [um] 横方向 (TIE設定に合わせて4.0に変更)
        self.z_pitch = 4.0      # [um] Z方向間隔 
        self.wavelength = 0.532 # [um] 緑色蛍光を想定
        self.bead_val = 65536.0 # ビーズの輝度 
        
    def fresnel_propagate(self, U_in, dist_z):
        """
        角スペクトル法によるフレネル伝搬 (MATLABコードのPython版)
        """
        if dist_z == 0:
            return U_in
        
        ny, nx = U_in.shape
        
        # 周波数座標の作成
        fx = np.fft.fftfreq(nx, d=self.pixel_size)
        fy = np.fft.fftfreq(ny, d=self.pixel_size)
        FX, FY = np.meshgrid(fx, fy)
        
        # 伝達関数 H = exp(-i * pi * lambda * z * (fx^2 + fy^2))
        H = np.exp(-1j * np.pi * self.wavelength * dist_z * (FX**2 + FY**2))
        
        # FFT -> 掛け算 -> IFFT
        U_out = ifft2(fft2(U_in) * H)
        return U_out

    def generate_sample(self):
        # 1. 空間の初期化 (32x32x128)
        # 論文では Z=128, X=32, Y=32
        label_vol_small = np.zeros(self.sim_shape, dtype=np.float32)
        raw_vol_small = np.zeros(self.sim_shape, dtype=np.float32)
        
        # 2. ビーズの配置 (1~10個) 
        num_beads = random.randint(1, 10)
        bead_positions = []
        
        for _ in range(num_beads):
            # ランダムな位置 (縁は避ける)
            bz = random.randint(10, self.sim_shape[0] - 11)
            by = random.randint(4, self.sim_shape[1] - 5)
            bx = random.randint(4, self.sim_shape[2] - 5)
            
            # Labelに配置
            label_vol_small[bz, by, bx] = self.bead_val
            bead_positions.append((bz, by, bx))
            
        # 3. 伝搬計算によるRawデータ生成 (コヒーレント伝搬)
        
        z_coords = (np.arange(self.sim_shape[0]) - self.sim_shape[0]//2) * self.z_pitch
        
        for target_z_idx in range(self.sim_shape[0]):
            # ターゲット層のZ位置
            z_target = z_coords[target_z_idx]
            
            # この層における強度の加算用 (インコヒーレント加算)
            intensity_accum = np.zeros((self.sim_shape[1], self.sim_shape[2]), dtype=np.float32)
            
            for (bz, by, bx) in bead_positions:
                z_source = z_coords[bz]
                dist = z_target - z_source
                
                # 点光源 (位相0と仮定)
                u_source = np.zeros((self.sim_shape[1], self.sim_shape[2]), dtype=np.complex64)
                u_source[by, bx] = np.sqrt(self.bead_val) 
                
                # 伝搬
                u_prop = self.fresnel_propagate(u_source, dist)
                
                # 強度を加算 (インコヒーレント)
                intensity_accum += np.abs(u_prop)**2
            
            raw_vol_small[target_z_idx, :, :] = intensity_accum

        # 4. リサイズ (バイリニア補間) 
        zoom_factors = (
            self.final_shape[0] / self.sim_shape[0], # Z: 128/128 = 1.0
            self.final_shape[1] / self.sim_shape[1], # Y: 128/32 = 4.0
            self.final_shape[2] / self.sim_shape[2]  # X: 128/32 = 4.0
        )
        
        label_vol_final = scipy.ndimage.zoom(label_vol_small, zoom_factors, order=1)
        raw_vol_final = scipy.ndimage.zoom(raw_vol_small, zoom_factors, order=1)
        
        # 正規化 (0.0 - 1.0)
        if raw_vol_final.max() > 0:
            raw_vol_final /= raw_vol_final.max()
        if label_vol_final.max() > 0:
            label_vol_final /= label_vol_final.max()
            
        label_vol_final[label_vol_final < 0.1] = 0

        # 次元拡張 (C, D, H, W) -> (1, 128, 128, 128)
        return raw_vol_final[np.newaxis, ...], label_vol_final[np.newaxis, ...]

def main():
    parser = argparse.ArgumentParser(description="Generate coherent diffraction dataset (Field Summation)")
    parser.add_argument("--outdir", type=str, default="D:/nosaka/data/coherent_propagation", help="Output directory")
    parser.add_argument("--train_count", type=int, default=800, help="Number of training samples")
    parser.add_argument("--val_count", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--test_count", type=int, default=100, help="Number of test samples")
    args = parser.parse_args()
    
    generator = PaperSimulationGenerator()
    
    splits = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count
    }
    
    for split, count in splits.items():
        if count == 0:
            continue
            
        save_dir = os.path.join(args.outdir, split)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Generating {split} set: {count} samples -> {save_dir}")
        
        for i in tqdm(range(count)):
            raw, label = generator.generate_sample()
            filename = f"sample_{i:04d}.h5"
            with h5py.File(os.path.join(save_dir, filename), 'w') as f:
                f.create_dataset("raw", data=raw, compression='gzip')
                f.create_dataset("label", data=label, compression='gzip')

    print("All generation completed.")

if __name__ == "__main__":
    main()
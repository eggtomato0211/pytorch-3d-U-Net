import numpy as np
import scipy.ndimage
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import h5py
import os
from tqdm import tqdm
import random

class PaperSimulationGenerator:
    def __init__(self):
        # 論文 3.2.1 学習データセットの記述に基づくパラメータ
        self.sim_shape = (128, 32, 32)  # (Z, Y, X) 計算時のサイズ 
        self.final_shape = (128, 128, 128) # 最終的なサイズ 
        
        # 光学パラメータ（論文や一般的実験条件からの推定）
        self.pixel_size = 1.0   # [um] 横方向 (32pxでビーズを表現するため仮定)
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
        # ※ 近軸近似フレネル回折の核
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
            # 修正: Pythonのrangeは終了値を含まないので、上限を調整
            # sim_shape[0] - 11 だと 128-11=117. randint(10, 117)
            bz = random.randint(10, self.sim_shape[0] - 11)
            by = random.randint(4, self.sim_shape[1] - 5)
            bx = random.randint(4, self.sim_shape[2] - 5)
            
            # Labelに配置
            label_vol_small[bz, by, bx] = self.bead_val
            bead_positions.append((bz, by, bx))
            
        # 3. 伝搬計算によるRawデータ生成 (TIEソルバーは使わず、直接伝搬)
        # "Numerical reconstruction gives us volume images... via Fresnel propagation" [cite: 18]
        # 各Z層について、全てのビーズからのボケ像を足し合わせる
        
        # Z座標の配列 (中心を0とする相対座標)
        z_coords = (np.arange(self.sim_shape[0]) - self.sim_shape[0]//2) * self.z_pitch
        
        for target_z_idx in range(self.sim_shape[0]):
            # ターゲット層のZ位置
            z_target = z_coords[target_z_idx]
            
            # この層における複素振幅の加算用 (インコヒーレントなので強度を加算)
            intensity_accum = np.zeros((self.sim_shape[1], self.sim_shape[2]), dtype=np.float32)
            
            for (bz, by, bx) in bead_positions:
                # ビーズのZ位置
                z_source = z_coords[bz]
                
                # 伝搬距離
                dist = z_target - z_source
                
                # 点光源を作る
                u_source = np.zeros((self.sim_shape[1], self.sim_shape[2]), dtype=np.complex64)
                u_source[by, bx] = np.sqrt(self.bead_val) # 強度が65536になるよう振幅を設定
                
                # 伝搬
                u_prop = self.fresnel_propagate(u_source, dist)
                
                # 強度を加算 (インコヒーレント)
                intensity_accum += np.abs(u_prop)**2
            
            raw_vol_small[target_z_idx, :, :] = intensity_accum

        # 4. リサイズ (バイリニア補間) 
        # 32x32x128 -> 128x128x128
        # Z方向(軸0)はサイズ変更なし、Y, X(軸1,2)を4倍にする
        zoom_factors = (
            self.final_shape[0] / self.sim_shape[0], # Z: 128/128 = 1.0
            self.final_shape[1] / self.sim_shape[1], # Y: 128/32 = 4.0
            self.final_shape[2] / self.sim_shape[2]  # X: 128/32 = 4.0
        )
        
        # order=1 はバイリニア(重線形)補間
        label_vol_final = scipy.ndimage.zoom(label_vol_small, zoom_factors, order=1)
        raw_vol_final = scipy.ndimage.zoom(raw_vol_small, zoom_factors, order=1)
        
        # 正規化 (0.0 - 1.0)
        if raw_vol_final.max() > 0:
            raw_vol_final /= raw_vol_final.max()
        if label_vol_final.max() > 0:
            label_vol_final /= label_vol_final.max()
            
        # 閾値処理 (Labelは補間でボケるので、二値化して戻すのもありだが、論文は言及なしのためそのままか)
        # ここでは視認性のため、Labelは少しクリーンにする
        label_vol_final[label_vol_final < 0.1] = 0

        # 次元拡張 (C, D, H, W) -> (1, 128, 128, 128)
        return raw_vol_final[np.newaxis, ...], label_vol_final[np.newaxis, ...]

# ==========================================
# 実行部
# ==========================================
def main():
    save_dir = "dataset_paper_repro"
    os.makedirs(save_dir, exist_ok=True)
    
    generator = PaperSimulationGenerator()
    
    print("論文の手法に基づいてデータを生成中...")
    # 例として10個生成
    for i in tqdm(range(10)):
        raw, label = generator.generate_sample()
        
        with h5py.File(os.path.join(save_dir, f"sample_{i:04d}.h5"), 'w') as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("label", data=label)
            
    print("完了しました。")

if __name__ == "__main__":
    main()

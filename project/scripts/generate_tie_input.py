import os
import random
import numpy as np
import h5py
from scipy.ndimage import zoom, gaussian_filter
from dataclasses import dataclass

# ==========================================
# 1. 設定管理クラス
# ==========================================
@dataclass
class SimConfig:
    # --- 保存設定 ---
    output_dir: str = "dataset_thesis_noise_robust"
    num_samples: int = 1000         # 必要に応じて変更してください
    
    # --- 空間解像度 ---
    # 論文仕様: 32x32x128 で作成し、128x128x128 に拡大
    sim_shape: tuple = (128, 32, 32)   # シミュレーション時の解像度 (Z, Y, X)
    final_shape: tuple = (128, 128, 128) # 学習時の入力解像度 (Z, Y, X)
    
    # --- 物理パラメータ ---
    wavelength: float = 532.0e-9  # 波長 (m)
    z_pitch: float = 4.0e-6       # Z軸方向の1画素の長さ
    dz_tie: float = 4.0e-6        # TIE撮影時の焦点ずらし幅
    target_pixel_size: float = 1.0e-6 # 最終画像(128px)での1画素サイズ
    
    # --- ビーズ(光源)の設定 ---
    num_beads_range: tuple = (1, 11) # 1以上11未満
    bead_intensity: float = 65536.0  # 16bit Max
    
    # --- 層の配置設定 ---
    layer_counts: tuple = (8, 16, 32, 64, 128) 
    
    # --- ノイズ設定 (重要: 修正済み) ---
    add_noise: bool = True
    poisson_scale: float = 3000.0
    # 修正: 16bit画像に対し0.05は小さすぎるため、実機相当(例:50.0)に変更
    gaussian_sigma: float = 50.0  
    
    # --- TIE計算の前処理 ---
    # ノイズによる微分の暴れを抑えるための平滑化
    tie_blur_sigma: float = 1.0 

    # --- 前処理 ---
    normalize_01: bool = True     # 学習用に0-1正規化を行う

# ==========================================
# 2. 物理計算コア関数
# ==========================================
def nearprop_conv(u_in, dx, dy, wavelength, z_dist):
    """フレネル伝搬計算 (Angular Spectrum Method)"""
    Ny, Nx = u_in.shape
    fx = np.fft.fftfreq(Nx, dx); fy = np.fft.fftfreq(Ny, dy)
    FX, FY = np.meshgrid(fx, fy)
    # バンドリミットを考慮する場合もあるが、ここではシンプルに実装
    phase = -1j * np.pi * wavelength * z_dist * (FX**2 + FY**2)
    H = np.exp(phase)
    return np.fft.ifft2(np.fft.fft2(u_in) * H)

def solve_tie_fft(I_minus, I_center, I_plus, dz, wavelength, dx, reg_param=0.5):
    """TIE位相回復ソルバ"""
    Ny, Nx = I_center.shape
    
    # 軸方向の強度微分 dI/dz
    dIdz = (I_plus - I_minus) / (2 * dz)
    
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dx)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    
    # 逆ラプラシアン (正則化付き)
    inv_laplacian = 1.0 / (K2 + reg_param)
    inv_laplacian[0, 0] = 0 
    
    # 式(2.7)に基づく解法
    tmp_psi = np.fft.ifft2(inv_laplacian * np.fft.fft2(dIdz))
    grad_psi_x = np.fft.ifft2(1j * KX * np.fft.fft2(tmp_psi))
    grad_psi_y = np.fft.ifft2(1j * KY * np.fft.fft2(tmp_psi))
    
    I_reg = I_center + 1e-6 # ゼロ除算防止
    field_x = grad_psi_x / I_reg
    field_y = grad_psi_y / I_reg
    
    div_field = np.fft.ifft2(1j * KX * np.fft.fft2(field_x) + 1j * KY * np.fft.fft2(field_y))
    result = np.fft.ifft2(inv_laplacian * np.fft.fft2(div_field))
    
    k0 = 2 * np.pi / wavelength
    phase = -k0 * np.real(result)
    return phase

def add_noise_func(image, cfg: SimConfig):
    """ノイズ付加関数"""
    if not cfg.add_noise: return image
    
    # Poisson Noise (ショットノイズ)
    img_safe = np.maximum(image, 1e-9)
    # スケールが大きいほどS/Nが良い。値を整数化してポアソン分布に従わせる
    noisy = np.random.poisson(img_safe / 65536.0 * cfg.poisson_scale) * (65536.0 / cfg.poisson_scale)
    
    # Gaussian Noise (読み出しノイズ等)
    gauss = np.random.normal(0, cfg.gaussian_sigma, image.shape)
    
    return np.maximum(noisy + gauss, 0)

# ==========================================
# 3. 論文仕様のラベル生成ロジック
# ==========================================
def generate_label_thesis_strict(cfg: SimConfig):
    """論文仕様に基づく厳密なラベル生成"""
    Nz, Ny, Nx = cfg.sim_shape
    vol = np.zeros(cfg.sim_shape, dtype=np.float32)
    
    # 1. 層の枚数を決定
    num_active_layers = random.choice(cfg.layer_counts)
    
    # 2. 配置モード決定 (等間隔 or ランダム)
    mode = random.choice(['equal', 'random'])
    active_z_indices = []
    
    if mode == 'equal':
        # 等間隔
        if num_active_layers >= Nz:
            active_z_indices = np.arange(Nz)
        else:
            step = Nz // num_active_layers
            active_z_indices = np.arange(0, Nz, step)
    else:
        # ランダム配置
        active_z_indices = np.sort(np.random.choice(range(Nz), num_active_layers, replace=False))
        
    # 3. 各層にビーズ配置 (1-10個)
    for z_idx in active_z_indices:
        num_beads = np.random.randint(cfg.num_beads_range[0], cfg.num_beads_range[1])
        for _ in range(num_beads):
            by = np.random.randint(2, Ny - 2)
            bx = np.random.randint(2, Nx - 2)
            vol[z_idx, by, bx] = cfg.bead_intensity
            
    return vol

# ==========================================
# 4. データ生成メインプロセス
# ==========================================
def generate_sample(cfg: SimConfig):
    # A. 画素サイズ計算
    sim_nz, sim_ny, sim_nx = cfg.sim_shape
    fin_nz, fin_ny, fin_nx = cfg.final_shape
    sim_dx = cfg.target_pixel_size * (fin_nx / sim_nx)
    
    # B. Label作成
    label_low = generate_label_thesis_strict(cfg)

    # C. 撮影シミュレーション
    I_minus = np.zeros((sim_ny, sim_nx), dtype=np.float32)
    I_center = np.zeros((sim_ny, sim_nx), dtype=np.float32)
    I_plus = np.zeros((sim_ny, sim_nx), dtype=np.float32)
    
    z_coords = (np.arange(sim_nz) - sim_nz // 2) * cfg.z_pitch
    
    # 光波伝搬計算により焦点面画像を作成
    for i in range(sim_nz):
        layer_img = label_low[i, :, :]
        if np.max(layer_img) < 1e-6: continue
            
        z_pos = z_coords[i]
        u_source = np.sqrt(layer_img) + 0j
        
        # 各面への距離
        d_m = -cfg.dz_tie - z_pos
        d_c = 0.0 - z_pos
        d_p = +cfg.dz_tie - z_pos
        
        # 強度を加算 (インコヒーレント結像近似)
        I_minus += np.abs(nearprop_conv(u_source, sim_dx, sim_dx, cfg.wavelength, d_m))**2
        I_center += np.abs(nearprop_conv(u_source, sim_dx, sim_dx, cfg.wavelength, d_c))**2
        I_plus += np.abs(nearprop_conv(u_source, sim_dx, sim_dx, cfg.wavelength, d_p))**2

    # ノイズ付加
    I_minus = add_noise_func(I_minus, cfg)
    I_center_noisy = add_noise_func(I_center, cfg) # 振幅回復用
    I_plus = add_noise_func(I_plus, cfg)

    # --- 【改良点】TIE計算の安定化 ---
    # ノイズを含んだ画像をそのまま差分すると高周波ノイズが爆発するため
    # TIEの位相計算に使う画像にはガウシアンフィルタをかける
    I_minus_smooth = gaussian_filter(I_minus, cfg.tie_blur_sigma)
    I_plus_smooth = gaussian_filter(I_plus, cfg.tie_blur_sigma)
    # 中心画像は分母に来るので平滑化した方が安全だが、今回は強度画像としてはnoisyなものを使う
    # ただし微分計算の整合性のため、位相計算用にはsmoothを使う
    I_center_smooth = gaussian_filter(I_center_noisy, cfg.tie_blur_sigma)

    # D. TIE位相回復 & 3D再構成
    # 位相回復には平滑化した画像を使用
    restored_phase = solve_tie_fft(I_minus_smooth, I_center_smooth, I_plus_smooth, 
                                   cfg.dz_tie, cfg.wavelength, sim_dx)
    
    # 振幅にはノイズが乗ったままの画像を使用（ここがDenoisingタスクの肝）
    # 位相情報と振幅情報を統合
    U_recon = np.sqrt(I_center_noisy) * np.exp(1j * restored_phase)
    
    # 逆伝搬または順伝搬でボリュームデータを再構成 (論文第2章 フレネル伝搬計算)
    input_low = np.zeros(cfg.sim_shape, dtype=np.float32)
    for i in range(sim_nz):
        U_z = nearprop_conv(U_recon, sim_dx, sim_dx, cfg.wavelength, z_coords[i])
        input_low[i, :, :] = np.abs(U_z)**2

    # E. 拡大 (バイリニア補間)
    factors = (fin_nz / sim_nz, fin_ny / sim_ny, fin_nx / sim_nx)
    label_high = zoom(label_low, factors, order=1) 
    input_high = zoom(input_low, factors, order=1)

    # F. 正規化 (Normalization) --- 【最重要修正箇所】 ---
    if cfg.normalize_01:
        # Label: 教師データは理想的なので、既知の最大値で割る
        label_high = label_high / cfg.bead_intensity
        
        # Input: スパイクノイズ対策のため、99.9%タイル値を最大値として扱う
        robust_max = np.percentile(input_high, 99.9)
        if robust_max < 1e-6: robust_max = 1.0 # 安全策
        
        # 上限をクリップしてから正規化
        input_high = np.clip(input_high, 0, robust_max)
        input_high = input_high / robust_max
        
    # NaNチェック
    if np.isnan(input_high).any():
        print("Warning: NaN detected, replacing with 0")
        input_high = np.nan_to_num(input_high)
    
    # 1チャンネル出力 (Channel, Z, Y, X)
    input_combined = np.expand_dims(input_high, axis=0)
    label_high = np.expand_dims(label_high, axis=0) # ラベルも次元を合わせておく方が無難

    return input_combined, label_high

# ==========================================
# 5. 実行ブロック
# ==========================================
if __name__ == "__main__":
    # 保存先ディレクトリ（環境に合わせて書き換えてください）
    base_dir = "D:/nosaka/data/data_multichannel_robust"
    
    # 生成設定: (ディレクトリ名, 生成数)
    phases = [
        ("train", 800),
        ("val", 200),
        ("test", 250)
    ]
    
    for phase, count in phases:
        output_dir = os.path.join(base_dir, phase)
        
        # 設定のインスタンス化
        config = SimConfig(
            output_dir = output_dir,
            num_samples = count,
            # 必要に応じてノイズレベルをここで上書き可能
            # gaussian_sigma = 50.0 
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"--- {phase} データセット生成開始 ---")
        print(f"保存先: {config.output_dir}")
        print(f"予定数: {config.num_samples}")
        
        for i in range(config.num_samples):
            try:
                inp, lbl = generate_sample(config)
                
                fname = os.path.join(config.output_dir, f"sample_{i:04d}.h5")
                with h5py.File(fname, 'w') as f:
                    f.create_dataset('raw', data=inp, compression='gzip')
                    f.create_dataset('label', data=lbl, compression='gzip')
                
                if (i+1) % 10 == 0:
                    print(f"Generated: {i+1}/{config.num_samples}")
            except Exception as e:
                print(f"Error at sample {i}: {e}")
                
    print("全プロセスの完了。")
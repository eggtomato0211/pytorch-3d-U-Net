import os
import random
import numpy as np
import h5py
from scipy.ndimage import zoom
from dataclasses import dataclass

# ==========================================
# 1. Config (Thesis Strict)
# ==========================================
@dataclass
class ThesisStrictConfig:
    # Output path
    output_dir: str = "dataset_thesis_strict_repro"
    num_samples: int = 1000
    
    # Dimensions
    init_shape: tuple = (128, 32, 32)   # (Z, Y, X)
    final_shape: tuple = (128, 128, 128)
    
    # Physics parameters
    z_pitch: float = 4.0e-6
    wavelength: float = 532.0e-9
    target_pixel_size: float = 1.0e-6
    
    # Bead Properties
    # "1x1 pixel bead, val 65536"
    bead_val: float = 65536.0
    num_beads_range: tuple = (1, 11)
    
    layer_counts: tuple = (8, 16, 32, 64, 128)
    
    # Preprocessing
    normalize_01: bool = True

# ==========================================
# 2. Physics Core
# ==========================================
def nearprop_conv(u_in, dx, dy, wavelength, z_dist):
    """Fresnel Propagation"""
    Ny, Nx = u_in.shape
    fx = np.fft.fftfreq(Nx, dx)
    fy = np.fft.fftfreq(Ny, dy)
    FX, FY = np.meshgrid(fx, fy)
    
    phase = -1j * np.pi * wavelength * z_dist * (FX**2 + FY**2)
    H = np.exp(phase)
    
    return np.fft.ifft2(np.fft.fft2(u_in) * H)

# ==========================================
# 3. Label & Beads Generation
# ==========================================
def generate_label_data(cfg: ThesisStrictConfig):
    Nz, Ny, Nx = cfg.init_shape
    vol = np.zeros(cfg.init_shape, dtype=np.float32)
    beads_list = []
    
    num_active_layers = random.choice(cfg.layer_counts)
    mode = random.choice(['equal', 'random'])
    
    if mode == 'equal':
        if num_active_layers >= Nz:
            active_z_indices = np.arange(Nz)
        else:
            step = Nz // num_active_layers
            active_z_indices = np.arange(0, Nz, step)
    else:
        active_z_indices = np.sort(np.random.choice(range(Nz), num_active_layers, replace=False))
        
    for z_idx in active_z_indices:
        num_beads = np.random.randint(cfg.num_beads_range[0], cfg.num_beads_range[1])
        for _ in range(num_beads):
            by = np.random.randint(0, Ny)
            bx = np.random.randint(0, Nx)
            vol[z_idx, by, bx] = cfg.bead_val
            beads_list.append((z_idx, by, bx))
            
    return vol, beads_list

# ==========================================
# 4. Main Generation (Strict Fluorescence Logic)
# ==========================================
def generate_thesis_sample(cfg: ThesisStrictConfig):
    # Setup
    init_nz, init_ny, init_nx = cfg.init_shape
    fin_nz, fin_ny, fin_nx = cfg.final_shape
    init_dx = cfg.target_pixel_size * (fin_nx / init_nx)
    
    # 1. Label
    label_low, beads_list = generate_label_data(cfg)
    
    # 2. Input Data (Defocus Simulation)
    # Use Incoherent Sum of Intensities (Fluorescence)
    input_low = np.zeros(cfg.init_shape, dtype=np.float32)
    z_coords = (np.arange(init_nz) - init_nz // 2) * cfg.z_pitch
    
    for (bz, by, bx) in beads_list:
        # Single bead source
        u_bead = np.zeros((init_ny, init_nx), dtype=np.complex64)
        u_bead[by, bx] = np.sqrt(cfg.bead_val) # Amplitudel
        
        bead_z_pos = z_coords[bz]
        
        for i in range(init_nz):
            target_z_pos = z_coords[i]
            dist = target_z_pos - bead_z_pos
            
            if dist == 0:
                input_low[i, :, :] += np.abs(u_bead)**2
            else:
                u_prop = nearprop_conv(u_bead, init_dx, init_dx, cfg.wavelength, dist)
                input_low[i, :, :] += np.abs(u_prop)**2

    # 3. Upsample
    factors = (fin_nz / init_nz, fin_ny / init_ny, fin_nx / init_nx)
    label_high = zoom(label_low, factors, order=1) 
    input_high = zoom(input_low, factors, order=1)
    
    # 4. Normalize
    if cfg.normalize_01:
        label_high = label_high / cfg.bead_val
        input_high = input_high / (np.max(input_high) + 1e-9)
        
    # Add Channel Dim
    input_combined = np.expand_dims(input_high, axis=0)
    label_high = np.expand_dims(label_high, axis=0)
    
    return input_combined, label_high

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    base_dir = "D:/nosaka/data/dataset_thesis_strict_repro"
    phases = [("train", 800), ("val", 200)]
    
    for phase, count in phases:
        output_dir = os.path.join(base_dir, phase)
        config = ThesisStrictConfig(output_dir=output_dir, num_samples=count)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"--- {phase} Output: {output_dir} ---")
        
        for i in range(config.num_samples):
            try:
                inp, lbl = generate_thesis_sample(config)
                fname = os.path.join(output_dir, f"sample_{i:04d}.h5")
                with h5py.File(fname, 'w') as f:
                    f.create_dataset('raw', data=inp, compression='gzip')
                    f.create_dataset('label', data=lbl, compression='gzip')
                
                if (i+1) % 50 == 0: print(f"Gen: {i+1}/{config.num_samples}")
            except Exception as e:
                print(f"Error {i}: {e}")
                
    print("Done (Strict Mode)")
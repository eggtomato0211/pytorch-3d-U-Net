param(
    [Parameter(Mandatory=$true)][string]$File,
    [string]$SaveDir = "D:/nosaka/plots",
    [string]$RawKey = "raw",
    [string]$LabelKey = "label",
    [int]$MaxPoints = 50000,
    [double]$RawPercentile = 99.5,
    [ValidateSet('auto','percentile','absolute')][string]$LabelMode = 'auto',
    [double]$LabelPercentile = 99.0,
    [double]$LabelAbsThresh = 0.5
)

$ErrorActionPreference = 'Stop'
Write-Host "[plot_3d] file=$File -> $SaveDir" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null

$env:PLOT3D_FILE = $File
$env:PLOT3D_SAVEDIR = $SaveDir
$env:PLOT3D_RAW_KEY = $RawKey
$env:PLOT3D_LABEL_KEY = $LabelKey
$env:PLOT3D_MAX_POINTS = [string]$MaxPoints
$env:PLOT3D_RAW_PERCENTILE = [string]$RawPercentile
$env:PLOT3D_LABEL_MODE = $LabelMode
$env:PLOT3D_LABEL_PERCENTILE = [string]$LabelPercentile
$env:PLOT3D_LABEL_ABS_THRESH = [string]$LabelAbsThresh

$code = @'
import os
import os.path as op
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_volume(h5_path, key):
    if not key:
        return None
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f"'{key}' not found in {h5_path}. Available keys: {list(f.keys())}")
        vol = f[key][:]
    vol = np.squeeze(vol)
    if vol.ndim == 4:
        print(f"[INFO] 4D volume detected {vol.shape}, using channel 0")
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Dataset '{key}' must be 3D after squeeze (or 4D with chan), got shape {vol.shape}")
    return vol

def select_points_from_volume(volume, mode='percentile', thresh=99.5, max_points=50000):
    v = volume.astype(np.float32)
    if mode == 'percentile':
        t = np.percentile(v, thresh)
        zz, yy, xx = np.where(v > t)
    elif mode == 'absolute':
        zz, yy, xx = np.where(v > thresh)
    else:
        raise ValueError("mode must be 'percentile' or 'absolute'")
    n = zz.size
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        zz, yy, xx = zz[idx], yy[idx], xx[idx]
    return zz, yy, xx

def plot_points_tripanel(points_list, titles, colors, shape_zyx, out_path):
    entries = [(p, t, c) for (p, t, c) in zip(points_list, titles, colors) if p is not None]
    cols = len(entries)
    if cols == 0:
        print('No valid panels to plot.')
        return
    fig = plt.figure(figsize=(7 * cols, 7))
    for i, (pts, title, color) in enumerate(entries, start=1):
        ax = fig.add_subplot(1, cols, i, projection='3d')
        z, y, x = pts
        if z.size > 0:
            ax.scatter(x, y, z, s=1.5, alpha=0.35, c=color)
        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (Depth)')
        ax.set_xlim(0, shape_zyx[2]); ax.set_ylim(0, shape_zyx[1]); ax.set_zlim(0, shape_zyx[0])
        ax.invert_zaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")

def main():
    h5 = os.environ['PLOT3D_FILE']
    out_dir = os.environ['PLOT3D_SAVEDIR']
    raw_key = os.environ.get('PLOT3D_RAW_KEY', 'raw')
    label_key = os.environ.get('PLOT3D_LABEL_KEY', 'label')
    max_points = int(os.environ.get('PLOT3D_MAX_POINTS', '50000'))
    raw_pct = float(os.environ.get('PLOT3D_RAW_PERCENTILE', '99.5'))
    label_mode = os.environ.get('PLOT3D_LABEL_MODE', 'auto')
    label_pct = float(os.environ.get('PLOT3D_LABEL_PERCENTILE', '99.0'))
    label_abs = float(os.environ.get('PLOT3D_LABEL_ABS_THRESH', '0.5'))

    base = op.splitext(op.basename(h5))[0]

    raw = None
    label = None
    try:
        raw = load_volume(h5, raw_key)
    except Exception as e:
        print(f"[WARN] Skipping raw: {e}")
    try:
        label = load_volume(h5, label_key) if label_key else None
    except Exception as e:
        print(f"[WARN] Skipping label: {e}")

    if raw is None and label is None:
        raise RuntimeError('Neither raw nor label could be loaded. Check keys and file.')

    shape_zyx = raw.shape if raw is not None else label.shape

    raw_pts = None
    if raw is not None:
        raw_pts = select_points_from_volume(raw, mode='percentile', thresh=raw_pct, max_points=max_points)

    label_pts = None
    if label is not None:
        mode = label_mode
        if mode == 'auto':
            uniq = np.unique(label)
            if label.dtype.kind in ('i','u') or (uniq.size <= 6 and set(uniq.tolist()).issubset({0,1})):
                mode = 'absolute'
                thresh = label_abs
            else:
                mode = 'percentile'
                thresh = label_pct
        elif mode == 'absolute':
            thresh = label_abs
        else:
            thresh = label_pct
        label_pts = select_points_from_volume(label, mode=mode, thresh=thresh, max_points=max_points)

    # Individual panels
    if raw_pts is not None:
        out_path = op.join(out_dir, f"{base}_raw_3d.png")
        plot_points_tripanel([raw_pts], ['Raw (top percentile)'], ['blue'], shape_zyx, out_path)
    if label_pts is not None:
        out_path = op.join(out_dir, f"{base}_label_3d.png")
        plot_points_tripanel([label_pts], ['Label (mask/thresholded)'], ['red'], shape_zyx, out_path)

    # Combined panels
    combo = []
    titles = []
    colors = []
    if raw_pts is not None:
        combo.append(raw_pts); titles.append('Raw'); colors.append('blue')
    if label_pts is not None:
        combo.append(label_pts); titles.append('Label'); colors.append('red')
    if combo:
        out_path = op.join(out_dir, f"{base}_raw_label_3panel.png")
        plot_points_tripanel(combo, titles, colors, shape_zyx, out_path)

    print('[Done] 3D plotting finished')

if __name__ == '__main__':
    main()
'@

$code | .venv\Scripts\python.exe -

import argparse
import csv
import glob
import os
from typing import Iterable, Tuple

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import ndimage


def list_pairs(pred_dir: str, gt_dir: str, pred_key='predictions', gt_key='label'):
    pred_files = []
    for ext in ('*.h5', '*.hdf5', '*.hdf', '*.hd5'):
        pred_files.extend(glob.glob(os.path.join(pred_dir, ext)))
    pred_files.sort()

    for pf in pred_files:
        base = os.path.splitext(os.path.basename(pf))[0]
        # strip common suffixes from prediction filenames to find GT counterpart
        for suf in ['_predictions', '_prediction', '_pred']:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break

        candidates = []
        for ext in ('*.h5', '*.hdf5', '*.hdf', '*.hd5'):
            # try normal and _misty suffix
            for suffix in ('', '_misty'):
                path = os.path.join(gt_dir, base + suffix + ext[1:])
                if os.path.exists(path):
                    candidates.append(path)
        # fallback: any file in gt_dir (if only one)
        if not candidates:
            any_gt = []
            for ext in ('*.h5', '*.hdf5', '*.hdf', '*.hd5'):
                any_gt.extend(glob.glob(os.path.join(gt_dir, ext)))
            if len(any_gt) == 1:
                candidates = [any_gt[0]]
        if not candidates:
            yield pf, None
        else:
            yield pf, candidates[0]


def compute_sharpness(vol: np.ndarray) -> float:
    lap = ndimage.laplace(vol.astype(np.float32))
    return float(np.var(lap))


def psnr3d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    data_range = float(np.max(a) - np.min(a)) or 1.0
    return float(peak_signal_noise_ratio(a, b, data_range=data_range))


def ssim3d(a: np.ndarray, b: np.ndarray) -> float:
    # compute per-slice SSIM and average
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 4:
        a = a.squeeze(0)
        b = b.squeeze(0)
    zs = a.shape[0]
    vals = []
    for z in range(zs):
        vals.append(structural_similarity(a[z], b[z], data_range=np.ptp(b[z]) or 1.0))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description='Evaluate 3D prediction volumes against GT')
    ap.add_argument('--pred_dir', required=True, help='Directory with predicted H5s (expects dataset "predictions")')
    ap.add_argument('--gt_dir', required=True, help='Directory with GT H5s (expects dataset "label")')
    ap.add_argument('--out_csv', default='D:/nosaka/outputs/metrics.csv')
    ap.add_argument('--pred_key', default='predictions')
    ap.add_argument('--gt_key', default='label')
    ap.add_argument('--denorm_pred', action='store_true',
                    help='Apply (pred+1)/2 clipping to [0,1] before metrics (useful if preds are in [-1,1])')
    args = ap.parse_args()

    rows = [('pred_file', 'gt_file', 'psnr', 'ssim', 'sharp_pred', 'sharp_gt')]
    matched = 0
    for pf, gf in list_pairs(args.pred_dir, args.gt_dir, args.pred_key, args.gt_key):
        if gf is None:
            print(f"[SKIP] No GT for {pf}")
            continue
        try:
            with h5py.File(pf, 'r') as fp, h5py.File(gf, 'r') as fg:
                pred = fp[args.pred_key][:]
                gt = fg[args.gt_key][:]
                if args.denorm_pred:
                    pred = np.clip((pred + 1.0) / 2.0, 0.0, 1.0)
                # squeeze possible channel dim
                pred = np.squeeze(pred)
                gt = np.squeeze(gt)
                if pred.shape != gt.shape:
                    print(f"[WARN] Shape mismatch pred={pred.shape} gt={gt.shape} for {pf}")
                psnr = psnr3d(gt, pred)
                ssim = ssim3d(gt, pred)
                sp = compute_sharpness(pred)
                sg = compute_sharpness(gt)
                rows.append((pf, gf, psnr, ssim, sp, sg))
                matched += 1
        except Exception as e:
            print(f"[ERR] {pf}: {e}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"Saved: {args.out_csv} (rows={len(rows)-1}, matched={matched})")


if __name__ == '__main__':
    raise SystemExit(main())

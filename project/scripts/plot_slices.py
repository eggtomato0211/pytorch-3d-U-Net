"""Compare prediction and ground truth on 2D slices with optional raw context.

Outputs a grid image with rows of slices at fixed depths: Raw | Pred | GT | |Pred-GT| heatmap.
"""

import argparse
import os
import os.path as op
from typing import List

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_h5(path: str, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"'{key}' not found in {path}. keys={list(f.keys())}")
        arr = np.squeeze(f[key][:])
    if arr.ndim != 3:
        raise ValueError(f"Dataset '{key}' must be 3D after squeeze, got {arr.shape}")
    return arr


def select_slices(depth: int, num: int = 4) -> List[int]:
    if depth <= num:
        return list(range(depth))
    idx = np.linspace(0, depth - 1, num=num, dtype=int)
    return list(idx)


def normalize_image(img: np.ndarray) -> np.ndarray:
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        return (img - vmin) / (vmax - vmin)
    return np.zeros_like(img, dtype=float)


def plot_grid(raw, pred, gt, out_path: str, num_slices: int = 4):
    depth = pred.shape[0]
    slices = select_slices(depth, num=num_slices)

    # rows = slices, cols = Raw | Pred | GT | Diff
    cols = [(raw, "Raw"), (pred, "Pred"), (gt, "GT"), (np.abs(pred - gt), "|Pred-GT|")]
    fig, axes = plt.subplots(len(slices), len(cols), figsize=(4 * len(cols), 3 * len(slices)))

    for r, z in enumerate(slices):
        for c, (vol, title) in enumerate(cols):
            ax = axes[r, c]
            img = vol[z]
            if title == "|Pred-GT|":
                im = ax.imshow(img, cmap="inferno")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(normalize_image(img), cmap="gray")
            ax.set_title(f"{title} z={z}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot slices: Raw | Pred | GT | Diff")
    ap.add_argument("--pred", required=True, help="H5 file with predictions")
    ap.add_argument("--gt", required=True, help="H5 file with ground truth (raw/label)")
    ap.add_argument("--out", default="D:/nosaka/plots/compare.png")
    ap.add_argument("--pred_key", default="predictions")
    ap.add_argument("--gt_key", default="label")
    ap.add_argument("--raw_key", default="raw", help="raw key in GT file for context; ignored if missing")
    ap.add_argument("--slices", type=int, default=4, help="number of slices to show")
    ap.add_argument("--denorm_pred", action="store_true",
                    help="Apply (pred+1)/2 and clip to [0,1] before plotting (use if preds are in [-1,1])")
    args = ap.parse_args()

    pred = load_h5(args.pred, args.pred_key)
    gt = load_h5(args.gt, args.gt_key)
    if args.denorm_pred:
        pred = np.clip((pred + 1.0) / 2.0, 0.0, 1.0)

    if pred.shape != gt.shape:
        print(f"[WARN] shape mismatch pred={pred.shape} gt={gt.shape}, proceeding")

    # raw is optional
    raw = None
    try:
        raw = load_h5(args.gt, args.raw_key)
    except Exception as e:
        print(f"[WARN] raw not loaded: {e}")

    # align shapes
    target_shape = gt.shape
    pred = np.array(pred)
    if pred.shape != target_shape:
        minz = min(pred.shape[0], target_shape[0])
        pred = pred[:minz]
        gt = gt[:minz]
        if raw is not None:
            raw = raw[:minz]

    if raw is None:
        raw = pred  # fallback

    os.makedirs(op.dirname(args.out) or ".", exist_ok=True)
    plot_grid(raw, pred, gt, args.out, num_slices=args.slices)


if __name__ == "__main__":
    main()

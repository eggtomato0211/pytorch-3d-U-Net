"""
Plot distributions and scatter of metric differences between two CSVs.

Usage:
    python project/scripts/plot_metric_diffs.py \
        --noisy_csv D:/nosaka/outputs/metrics.csv \
        --clean_csv D:/nosaka/outputs/clean/metrics_clean.csv \
        --out_dir D:/nosaka/outputs/plots

The script expects the CSVs produced by eval_metrics.py, i.e. columns:
pred_file, gt_file, psnr, ssim, sharp_pred, sharp_gt
"""

import argparse
import os
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def basename(series: pd.Series) -> pd.Series:
    return series.str.extract(r"([^\\/]+)\.h5")[0]


def load_and_merge(noisy_csv: str, clean_csv: str) -> pd.DataFrame:
    noisy = pd.read_csv(noisy_csv)
    clean = pd.read_csv(clean_csv)
    noisy["base"] = basename(noisy["gt_file"])
    clean["base"] = basename(clean["gt_file"])
    merged = noisy.merge(clean, on="base", suffixes=("_noisy", "_clean"))
    merged["d_psnr"] = merged["psnr_clean"] - merged["psnr_noisy"]
    merged["d_ssim"] = merged["ssim_clean"] - merged["ssim_noisy"]
    return merged


def plot_hist(data: pd.Series, title: str, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=40, alpha=0.8, color="#1f77b4")
    plt.axvline(data.mean(), color="r", linestyle="--", label=f"mean={data.mean():.3f}")
    plt.axvline(data.median(), color="g", linestyle=":", label=f"median={data.median():.3f}")
    plt.title(title)
    plt.xlabel("difference")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(df["psnr_noisy"], df["psnr_clean"], alpha=0.4, s=20)
    lims = [
        min(df["psnr_noisy"].min(), df["psnr_clean"].min()),
        max(df["psnr_noisy"].max(), df["psnr_clean"].max()),
    ]
    plt.plot(lims, lims, "k--", label="y=x")
    plt.xlabel("PSNR noisy model")
    plt.ylabel("PSNR clean model")
    plt.title("PSNR comparison (noisy vs clean model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot metric differences between two CSVs or a precomputed diff CSV.")
    ap.add_argument("--noisy_csv")
    ap.add_argument("--clean_csv")
    ap.add_argument("--diff_csv", default="D:/nosaka/outputs/clean_vs_noisy.csv",
                    help="Optional: CSV already containing psnr_noisy/psnr_clean/... (as produced by comparison script)")
    ap.add_argument("--out_dir", default="D:/nosaka/outputs/plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.diff_csv:
        merged = pd.read_csv(args.diff_csv)
    else:
        if not args.noisy_csv or not args.clean_csv:
            raise SystemExit("Provide either --diff_csv or both --noisy_csv and --clean_csv")
        merged = load_and_merge(args.noisy_csv, args.clean_csv)

    hist_psnr_path = op.join(args.out_dir, "diff_psnr_hist.png")
    hist_ssim_path = op.join(args.out_dir, "diff_ssim_hist.png")
    scatter_psnr_path = op.join(args.out_dir, "psnr_scatter.png")

    plot_hist(merged["d_psnr"], "ΔPSNR (clean - noisy)", hist_psnr_path)
    plot_hist(merged["d_ssim"], "ΔSSIM (clean - noisy)", hist_ssim_path)
    plot_scatter(merged, scatter_psnr_path)

    summary_path = op.join(args.out_dir, "diff_summary.txt")
    with open(summary_path, "w") as f:
        f.write("ΔPSNR stats:\n")
        f.write(str(merged["d_psnr"].describe()))
        f.write("\n\nΔSSIM stats:\n")
        f.write(str(merged["d_ssim"].describe()))
    print(f"Saved histograms and scatter to {args.out_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

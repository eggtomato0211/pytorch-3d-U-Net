import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# --- Configuration ---
CSV_PATH = Path(r"D:/nosaka/outputs/metrics.csv")
OUTPUT_PLOT = Path(r"D:/nosaka/outputs/metrics_plots.png")

def extract_metadata(filepath):
    """
    Extracts number of images, bead count (plots), and randomness from the filename.
    Example: 128images_10plots_fixed_randomFalse_NumberFrom1_predictions.h5
    """
    filename = os.path.basename(filepath)
    
    # Extract image count
    img_match = re.search(r"(\d+)images", filename)
    images = int(img_match.group(1)) if img_match else 0
    
    # Extract bead count (plots)
    plot_match = re.search(r"(\d+)plots", filename)
    beads = int(plot_match.group(1)) if plot_match else 0
    
    # Extract randomness
    random_match = re.search(r"random(True|False)", filename)
    is_random = random_match.group(1) if random_match else "Unknown"
    
    return images, beads, is_random

def main():
    if not CSV_PATH.exists():
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return

    print(f"Reading metrics from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Metadata extraction
    metadata = df['pred_file'].apply(extract_metadata)
    df[['Images', 'Beads', 'Random']] = pd.DataFrame(metadata.tolist(), index=df.index)
    
    # Setup plots (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Model Performance Dashboard (Misty Data)", fontsize=22, fontweight='bold')

    # 1. Impact of Density on SSIM (Scatter + Mean)
    ax1 = axes[0, 0]
    ssim_means = df.groupby('Beads')['ssim'].mean()
    ax1.scatter(df['Beads'], df['ssim'], color='gray', alpha=0.3, label='Individual Samples')
    ax1.plot(ssim_means.index, ssim_means.values, 'bo-', linewidth=2, label='Mean SSIM')
    ax1.set_title("SSIM vs Bead Density", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Bead Count (Density)")
    ax1.set_ylabel("SSIM")
    ax1.set_xticks(range(1, 11))
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Impact of Density on PSNR (Scatter + Mean)
    ax2 = axes[0, 1]
    psnr_means = df.groupby('Beads')['psnr'].mean()
    ax2.scatter(df['Beads'], df['psnr'], color='gray', alpha=0.3, label='Individual Samples')
    ax2.plot(psnr_means.index, psnr_means.values, 'ro-', linewidth=2, label='Mean PSNR')
    ax2.set_title("PSNR vs Bead Density", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Bead Count (Density)")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_xticks(range(1, 11))
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. SSIM Distribution (Histogram)
    ax3 = axes[1, 0]
    ax3.hist(df['ssim'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
    ax3.set_title("SSIM Distribution", fontsize=16, fontweight='bold')
    ax3.set_xlabel("SSIM")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 4. PSNR Distribution (Histogram)
    ax4 = axes[1, 1]
    ax4.hist(df['psnr'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.set_title("PSNR Distribution", fontsize=16, fontweight='bold')
    ax4.set_xlabel("PSNR (dB)")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"✅ Dashboard saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()

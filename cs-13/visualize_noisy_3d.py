import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ==========================================
# ⚙️ 設定パラメータ (完全版)
# ==========================================
# 1. パス設定
DATA_DIR = Path(r"D:/nosaka/data/3d-holography_output/Train")
NOISY_DIR = Path(r"D:/nosaka/data/3d-holography_output/Train_misty")
SAVE_DIR = Path(r"c:\Users\Owner\mizusaki\pytorch-3dunet\cs-13")

# 2. 表示ファイル設定
FILE_INDEX = 0  # フォルダ内の何番目のファイルを表示するか

# 3. プロット設定
MAX_POINTS = 50000  # 表示する点の最大数

# 4. 閾値係数
# 平均値の何倍を「信号」とするか (Original/Misty共通)
SIGNAL_THRESHOLD_FACTOR = 2.0
# 平均値の何倍を「霧」の最低ラインとするか (Mistyのみ適用)
MIST_THRESHOLD_FACTOR = 0.1
# ==========================================


def plot_split_colors(ax, vol, min_val, boundary_val, max_points, show_haze=True):
    """
    データを「霧(青)」と「信号(黄)」に分けてプロットする関数
    
    Args:
        min_val: これより小さい値は無視 (足切り)
        boundary_val: これより大きい値は「信号」、小さい値は「霧」
        show_haze: Falseの場合、boundary_val未満の「霧」を描画しない
    """
    
    # -----------------------------------------------------------
    # 1. 霧パート (Blue/Cyan) - 弱い信号
    # -----------------------------------------------------------
    if show_haze:
        # 範囲: min_val 〜 boundary_val
        z1, y1, x1 = np.where((vol >= min_val) & (vol < boundary_val))
        
        if len(z1) > 0:
            # 点が多すぎる場合は間引く
            if len(z1) > max_points:
                idx = np.random.choice(len(z1), max_points, replace=False)
                z1, y1, x1 = z1[idx], y1[idx], x1[idx]
            
            vals1 = vol[z1, y1, x1]
            
            # 0-1に正規化 (Winterカラーマップ用)
            norm1 = (vals1 - min_val) / (boundary_val - min_val + 1e-6)
            colors1 = plt.cm.winter(norm1)

            # ★変更点: 霧を「多めに・濃く」見せる設定
            # 透明度: 0.2(20%) 〜 0.8(80%)
            colors1[:, 3] = 0.2 + 0.6 * norm1 
            # サイズ: 2 (少し大きく)
            haze_size = 2
            
            ax.scatter(x1, y1, z1, c=colors1, s=haze_size, linewidth=0, label='Haze (Mist)')

    # -----------------------------------------------------------
    # 2. 信号パート (Red/Yellow) - 強い信号
    # -----------------------------------------------------------
    # 範囲: boundary_val 以上
    z2, y2, x2 = np.where(vol >= boundary_val)
    
    if len(z2) > 0:
        if len(z2) > max_points // 2:
            idx = np.random.choice(len(z2), max_points // 2, replace=False)
            z2, y2, x2 = z2[idx], y2[idx], x2[idx]

        vals2 = vol[z2, y2, x2]
        
        # 0-1に正規化 (Hotカラーマップ用)
        norm2 = (vals2 - boundary_val) / (vol.max() - boundary_val + 1e-6)
        colors2 = plt.cm.hot(norm2)
        
        # 信号はくっきり不透明に
        colors2[:, 3] = 0.8 + 0.2 * norm2 
        
        ax.scatter(x2, y2, z2, c=colors2, s=2, linewidth=0, label='Signal (Data)')


def main():
    # ファイル検索
    noisy_files = sorted(list(NOISY_DIR.glob("*_misty.h5")))
    if not noisy_files:
        print(f"❌ ファイルが見つかりません: {NOISY_DIR}")
        return

    # インデックス処理
    idx = FILE_INDEX if 0 <= FILE_INDEX < len(noisy_files) else 0
    noisy_path = noisy_files[idx]
    
    # 元ファイル名の特定 (_misty を削除)
    original_name = noisy_path.name.replace("_misty.h5", ".h5")
    original_path = DATA_DIR / original_name
    
    if not original_path.exists():
        print(f"❌ 元ファイルが見つかりません: {original_name}")
        return

    print(f"👀 可視化対象: {original_name}")

    # データの読み込み
    with h5py.File(original_path, "r") as f: original_vol = f["raw"][:]
    with h5py.File(noisy_path, "r") as f: noisy_vol = f["raw"][:]

    # 閾値の計算 (元データの平均値を基準にする)
    data_mean = np.mean(original_vol)
    
    # 信号ライン (これ以上は黄色)
    signal_boundary = data_mean * SIGNAL_THRESHOLD_FACTOR
    # 霧ライン (これ以上は青色)
    mist_min = data_mean * MIST_THRESHOLD_FACTOR

    print(f"📊 閾値設定:")
    print(f"  Signal Boundary (Yellow): {signal_boundary:.4f}")
    print(f"  Mist Minimum    (Blue)  : {mist_min:.4f}")

    # --- 3Dプロット作成 ---
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"Visuzalization: {original_name}", fontsize=14, fontweight='bold')

    # 1. 左側: Original (霧なし)
    ax1 = fig.add_subplot(121, projection='3d')
    plot_split_colors(ax1, original_vol, 
                      min_val=mist_min, 
                      boundary_val=signal_boundary, 
                      max_points=MAX_POINTS, 
                      show_haze=False) # ★重要: 元データは霧を表示しない
    ax1.set_title('Original Data (Signal Only)', fontweight='bold')

    # 2. 右側: Misty (霧あり)
    ax2 = fig.add_subplot(122, projection='3d')
    plot_split_colors(ax2, noisy_vol, 
                      min_val=mist_min, 
                      boundary_val=signal_boundary, 
                      max_points=MAX_POINTS, 
                      show_haze=True) # ★重要: こちらは霧を表示する
    ax2.set_title('Misty Data (Signal + Enhanced Haze)', fontweight='bold')

    # 共通設定 (視点・軸)
    for ax in [ax1, ax2]:
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-30)
        
        # 軸範囲を元データに合わせる
        ax.set_xlim(0, original_vol.shape[2])
        ax.set_ylim(0, original_vol.shape[1])
        ax.set_zlim(0, original_vol.shape[0])
        ax.invert_zaxis()

    ax2.legend(loc='upper right')

    # 保存
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / "check_misty_final.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 画像を保存しました: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()
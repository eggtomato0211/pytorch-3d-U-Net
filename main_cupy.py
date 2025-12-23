import time
from image_loader import ImageLoader
from propagation import Propagation
from phase_generator import PhaseGenerator
from hdf_manager import HDFManager
from reconstruction_processor import ReconstructionProcessor

# パラメータの設定
wav_len = 532.0 * 10**-9
Nx, Ny = 32, 32
dx = 3.45 * 10**-6
dy = dx
dz = 4 * 10**-6
pixels = 1
depthlevel = 128
# バッチ設定 (Total 1500 samples: 15 batches * 10 channels/batch * 10 conditions)
num_batches = 15
channels_per_batch = 10

number_of_plots = 10

 # `random_mode`がTrueならランダム、Falseなら等間隔のdepthsを生成
random_mode = True

# plot数がrandomかどうか
random_plots_mode = False

# 位相をランダムに生成するかどうか
random_phase_mode = False

# バッチループ
for batch in range(num_batches):
    print(f"--- Batch {batch+1}/{num_batches} ---")
    
    for random_mode in [True, False]:
        for image_number in [8, 16, 32, 64, 128]:
            mode_prefix = "random_" if random_mode else ""
            plots_mode_prefix = "randomPlots_" if random_plots_mode else ""

            # ディレクトリの作成
            # random_mode や random_plots_mode に基づき hdf_folder の先頭に `random_` を追加
            hdf_folder = f'D:/nosaka/data/3d-holography_output/{mode_prefix}{plots_mode_prefix}{Nx}x{Ny}x{depthlevel}_d={dz}_pixels={pixels}_{number_of_plots}plots_{image_number}images'

            image_folder = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\{plots_mode_prefix}{Nx}x{Ny}x{depthlevel}_1pxx{number_of_plots}'
            
            # 画像の読み込み (再シャッフルされるため毎回異なる画像が選ばれる)
            loader = ImageLoader(Nx, Ny, pixels, number_of_plots, channels_per_batch, depthlevel, folder_path=image_folder)
            images = loader.load_images()

            # 16bitから0~1の範囲に正規化
            images = [img / 255 for img in images]
            
            # 波面の伝搬クラス
            propagation = Propagation(wav_len, dx, dy)

            # 位相生成クラス
            phase_gen = PhaseGenerator(random_phase_mode)

            # 再構成処理クラス
            reconstructor = ReconstructionProcessor(Nx, Ny, dx, dy, dz, depthlevel, channels_per_batch, images, random_mode, phase_gen, propagation)

            # 各チャンネルごとに再構成を行い、HDFに保存
            for channel in range(channels_per_batch):
                # IDオフセット計算: Batch * Channels + Channel
                global_channel_id = batch * channels_per_batch + channel
                
                raw_data, label_data = reconstructor.process_channel(channel, image_number)
                hdf_manager = HDFManager(128, 128, depthlevel, dz, hdf_folder)
                # ファイル番号を一意にする
                hdf_manager.save_to_hdf(raw_data, label_data, f"{number_of_plots}plots_{image_number}images_{random_mode}randomMode_NumberFrom{global_channel_id * depthlevel + 1}.h5")

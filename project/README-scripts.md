Project scripts (safe wrappers)

- project/scripts/run_train.ps1: Runs upstream trainer unchanged.
  - Usage: `./project/scripts/run_train.ps1 [-Config project/configs/train_config.yaml]`

- project/scripts/run_predict.ps1: Runs upstream predictor unchanged.
  - Defaults: model to `D:/nosaka/checkpoint/best_checkpoint.pytorch`, output to `D:/nosaka/outputs/predict`.
  - Usage: `./project/scripts/run_predict.ps1 [-Config ...] [-ModelPath ...] [-OutputDir ...] [-Files file1.h5,file2.h5]`

- project/scripts/check_val_data.py: Checks that val/test file_paths exist, contain required HDF5 keys, and that patch_shape fits.
  - Usage: `python project/scripts/check_val_data.py --config project/configs/train_config.yaml --phase val`

- project/scripts/plot_3d.ps1: Self-contained 3D scatter plotter for HDF5 (no external dependency).
  - Usage: `./project/scripts/plot_3d.ps1 -File D:/nosaka/data/val/example.h5`
  - Options: `-SaveDir`, `-RawKey`, `-LabelKey`, `-RawPercentile`, `-LabelMode auto|percentile|absolute`, `-LabelPercentile`, `-LabelAbsThresh`, `-MaxPoints`

- project/scripts/plot_slices.py: Slice-wise comparison (Raw | Pred | GT | |Pred-GT| heatmap).
  - Usage: `python project/scripts/plot_slices.py --pred D:/nosaka/outputs/predict/Number1_predictions.h5 --gt C:/Users/Owner/Desktop/test250/Number1.h5 --out D:/nosaka/plots/Number1_compare.png`
  - Options: `--pred_key predictions --gt_key label --raw_key raw --slices 4`

- project/scripts/make_noisy.ps1: Wraps cs-13/generate_noisy_data.py with D: defaults.
  - Usage: `./project/scripts/make_noisy.ps1 -InDir D:/nosaka/data/train -OutDir D:/nosaka/data/noisy -TargetPSNR 28`

- project/scripts/validate_h5.py: Standalone H5 validator for keys and shapes.
  - Usage: `python project/scripts/validate_h5.py --paths D:/nosaka/data/val --require-label --min-shape 64 64 64`

- project/scripts/eval_metrics.py: Aggregates PSNR/SSIM/sharpness into CSV.
  - Usage: `python project/scripts/eval_metrics.py --pred_dir D:/nosaka/outputs/predict --gt_dir D:/nosaka/data/val --out_csv D:/nosaka/outputs/metrics.csv`

- project/scripts/compare_models.ps1: Predicts with two checkpoints and writes separate metrics CSVs.
  - Usage: `./project/scripts/compare_models.ps1 -Config project/configs/train_config.yaml -ModelA D:/nosaka/checkpoint/A.pth -ModelB D:/nosaka/checkpoint/B.pth -DataDir D:/nosaka/data/val`

Notes
- These scripts do not modify upstream pytorch3dunet code.
- Ensure your venv is activated before running: `& .\.venv\Scripts\Activate.ps1`.

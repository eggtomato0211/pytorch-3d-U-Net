param(
    [string]$Config = "project/configs/train_config.yaml",
    [string]$ModelPath = "D:/nosaka/checkpoint/best_checkpoint.pytorch",
    [string]$OutputDir = "D:/nosaka/outputs/predict", # kept for display; YAML controls actual output
    [string[]]$Files
)

$ErrorActionPreference = 'Stop'
Write-Host "[run_predict] Using config: $Config" -ForegroundColor Cyan
Write-Host "[run_predict] Model: $ModelPath" -ForegroundColor Cyan
Write-Host "[run_predict] OutputDir (YAML loaders.output_dir or override): $OutputDir" -ForegroundColor Cyan

$argsList = @('--config', $Config, '--model_path', $ModelPath)
if ($OutputDir) {
    # Now supported because YAML defines loaders.output_dir
    $argsList += @('--loaders.output_dir', $OutputDir)
}
if ($Files -and $Files.Count -gt 0) {
    $argsList += @('--loaders.test.file_paths') + $Files
    Write-Host "[run_predict] Test files: $($Files -join ', ')"
}

python -m pytorch3dunet.predict @argsList

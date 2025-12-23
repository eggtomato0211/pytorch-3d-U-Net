param(
    [string]$InDir = "D:/nosaka/data/train",
    [string]$OutDir = "D:/nosaka/data/noisy",
    [double]$TargetPSNR = 28.0
)

$ErrorActionPreference = 'Stop'
Write-Host "[make_noisy] In=$InDir Out=$OutDir TargetPSNR=$TargetPSNR" -ForegroundColor Cyan

$env:PYTHONUNBUFFERED = '1'

$env:MAKE_NOISY_IN = $InDir
$env:MAKE_NOISY_OUT = $OutDir
$env:MAKE_NOISY_PSNR = [string]$TargetPSNR

python cs-13/generate_noisy_data.py

param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$ModelA,
    [Parameter(Mandatory=$true)][string]$ModelB,
    [string]$DataDir = "D:/nosaka/data/val",
    [string]$OutRoot = "D:/nosaka/outputs/compare",
    [string]$MetricCsv = "D:/nosaka/outputs/compare/metrics.csv"
)

$ErrorActionPreference = 'Stop'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$OutA = Join-Path $OutRoot ("A_" + $ts)
$OutB = Join-Path $OutRoot ("B_" + $ts)

Write-Host "[compare] Predict A -> $OutA" -ForegroundColor Cyan
./project/scripts/run_predict.ps1 -Config $Config -ModelPath $ModelA -OutputDir $OutA -Files $DataDir

Write-Host "[compare] Predict B -> $OutB" -ForegroundColor Cyan
./project/scripts/run_predict.ps1 -Config $Config -ModelPath $ModelB -OutputDir $OutB -Files $DataDir

Write-Host "[compare] Eval metrics -> $MetricCsv" -ForegroundColor Cyan
python project/scripts/eval_metrics.py --pred_dir $OutA --gt_dir $DataDir --out_csv $MetricCsv
python project/scripts/eval_metrics.py --pred_dir $OutB --gt_dir $DataDir --out_csv ($MetricCsv -replace '\.csv$', '_B.csv')

Write-Host "[compare] Done"


param(
    [string]$Config = "project/configs/train_config.yaml"
)

$ErrorActionPreference = 'Stop'
Write-Host "[run_train] Using config: $Config"

python -m pytorch3dunet.train --config $Config


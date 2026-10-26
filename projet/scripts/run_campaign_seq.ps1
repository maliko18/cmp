$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "run_campaign.py"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $script --device seq @args
} else {
    & python $script --device seq @args
}
exit $LASTEXITCODE

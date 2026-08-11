# WhisperFlow-local installer (Windows)
# Run from this folder:  powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Host "Python 3.10+ is required: https://www.python.org/downloads/"; exit 1 }

Write-Host "Installing WhisperFlow-local ..."
python -m pip install --upgrade pip --quiet
python -m pip install "$PSScriptRoot"

# NVIDIA GPU? add the CUDA runtime libraries
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "NVIDIA GPU detected - installing CUDA libraries ..."
    python -m pip install "$PSScriptRoot[cuda]"
} else {
    Write-Host "No NVIDIA GPU detected - will run on CPU (edit config: smaller model recommended)."
}

Write-Host ""
Write-Host "Done. Start with:  whisperflow"
Write-Host "List audio devices:  whisperflow --list"
Write-Host "Config: ~\.whisperflow\config.json (created on first run)"

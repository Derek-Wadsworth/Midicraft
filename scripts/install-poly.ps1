# Install polyphonic mode deps (basic-pitch has a stale tensorflow pin on PyPI)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing polyphonic dependencies..."
pip install -r requirements-poly.txt
pip install basic-pitch==0.4.0 --no-deps
Write-Host "Done. Polyphonic mode (--mode poly) is ready."

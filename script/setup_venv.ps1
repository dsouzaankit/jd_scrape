# Creates .venv next to jd_scrape (parent of this script directory).
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$VenvPython = Join-Path $VenvRoot ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Path $VenvRoot -Force | Out-Null
Copy-Item -Force (Join-Path $ScriptDir "requirements-vector.txt") (Join-Path $VenvRoot "requirements-vector.txt")

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv at $VenvRoot\.venv ..."
    & python -m venv (Join-Path $VenvRoot ".venv")
}

& $VenvPython -m pip install -U pip
& (Join-Path $VenvRoot ".venv\Scripts\pip.exe") install -r (Join-Path $VenvRoot "requirements-vector.txt")
Write-Host "Done. Activate: & `"$VenvRoot\.venv\Scripts\Activate.ps1`""
$RepoRoot = Split-Path $ScriptDir -Parent
$ChromaArchived = Join-Path $RepoRoot "archive\chromadb\embed_staging_jd.py"
Write-Host "Archived Chroma embed (install: pip install -r `"$(Join-Path $RepoRoot 'archive\chromadb\requirements-chroma.txt')`"):"
Write-Host "  & `"$VenvPython`" `"$ChromaArchived`" --reset"
Write-Host "Run embed (DuckDB): & `"$VenvPython`" `"$ScriptDir\embed_staging_jd_duckdb.py`" --reset"

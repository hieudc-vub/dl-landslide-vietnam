param([ValidateSet('cuda','cpu')][string]$Compute = 'cuda')
# Use a dedicated environment; never install into the author's original environment.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program $Arguments" }
}
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Python launcher not found. Install Python 3.12 from python.org, then open a new PowerShell window.'
    }
    Invoke-Checked 'py' @('-3.12','-c','import sys; print(sys.version)')
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        Invoke-Checked 'py' @('-3.12','-m','venv','.venv')
    }
    $ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    Invoke-Checked $ProjectPython @('-m','pip','install','--upgrade','pip')
    $TorchIndex = if ($Compute -eq 'cuda') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
    Invoke-Checked $ProjectPython @('-m','pip','install','torch==2.9.0','torchvision==0.24.0','--index-url',$TorchIndex)
    Invoke-Checked $ProjectPython @('-m','pip','install','-e','.')
    Invoke-Checked $ProjectPython @('-m','pip','check')
    Invoke-Checked $ProjectPython @('-m','landslide','check-env')
    Write-Host 'SETUP COMPLETE. Next: run the smoke test in README.md.' -ForegroundColor Green
} catch {
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Read docs/troubleshooting.md. Existing datasets and results have not been removed.'
    exit 1
}

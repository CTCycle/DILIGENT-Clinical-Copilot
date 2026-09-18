[CmdletBinding()]
param(
    [ValidateSet('Full', 'LiveProvider')]
    [string]$Suite = 'Full'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$serverDir = Join-Path $repoRoot 'app/server'
$clientDir = Join-Path $repoRoot 'app/client'
$cacheRoot = Join-Path $repoRoot 'runtimes/cache'
$pytestCacheDir = Join-Path $cacheRoot 'pytest'
$python = Join-Path $serverDir '.venv/Scripts/python.exe'
$logDir = Join-Path $pytestCacheDir 'browser-e2e-logs'
$backendOut = Join-Path $logDir 'backend.out.log'
$backendErr = Join-Path $logDir 'backend.err.log'
$frontendOut = Join-Path $logDir 'frontend.out.log'
$frontendErr = Join-Path $logDir 'frontend.err.log'
$backend = $null
$frontend = $null
$backendReady = $false
$frontendReady = $false

function Write-Diagnostics([string]$label, $process, [string]$stdoutPath, [string]$stderrPath) {
    Write-Output "[$label]"
    if ($null -eq $process) {
        Write-Output 'Process was not created.'
    }
    else {
        $process.Refresh()
        Write-Output "PID=$($process.Id) HasExited=$($process.HasExited)"
        if ($process.HasExited) { Write-Output "ExitCode=$($process.ExitCode)" }
    }
    foreach ($path in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $path) {
            Write-Output "--- $path ---"
            Get-Content -LiteralPath $path -Tail 200
        }
    }
}

function Wait-HttpReady([string]$uri, [int]$timeoutSeconds = 180) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing $uri -TimeoutSec 3
            if ($response.StatusCode -lt 500) { return $true }
        }
        catch {
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Stop-ProcessTree($process) {
    if ($null -eq $process) { return }
    $process.Refresh()
    if (-not $process.HasExited) {
        & taskkill.exe /PID $process.Id /T /F *> $null
    }
}

New-Item -ItemType Directory -Path @($cacheRoot, $pytestCacheDir, $logDir) -Force | Out-Null
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:PIP_CACHE_DIR = Join-Path $cacheRoot 'pip'
$env:NPM_CONFIG_CACHE = Join-Path $cacheRoot 'npm'
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $cacheRoot 'playwright'
$env:HF_HOME = Join-Path $cacheRoot 'huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_ASSETS_CACHE = Join-Path $env:HF_HOME 'assets'
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:RUFF_CACHE_DIR = Join-Path $cacheRoot 'ruff'
$env:MYPY_CACHE_DIR = Join-Path $cacheRoot 'mypy'
$env:COVERAGE_FILE = Join-Path $cacheRoot 'coverage/.coverage'
$env:PYTHONPYCACHEPREFIX = Join-Path $cacheRoot 'python'
$env:CARGO_TARGET_DIR = Join-Path $cacheRoot 'cargo/target'
$env:PYTEST_ADDOPTS = "--basetemp=$pytestCacheDir/basetemp -o cache_dir=$pytestCacheDir"
$backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '7690' -WorkingDirectory $serverDir -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -WindowStyle Hidden -PassThru
try {
    if (-not (Wait-HttpReady 'http://127.0.0.1:7690/api/health') -or -not (Wait-HttpReady 'http://127.0.0.1:7690/api/model-config')) {
        Write-Diagnostics 'backend readiness failure' $backend $backendOut $backendErr
        throw 'Backend did not become ready before the browser E2E deadline.'
    }
    $backendReady = $true

    $frontend = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'preview', '--', '--host', '127.0.0.1', '--port', '9847', '--strictPort' -WorkingDirectory $clientDir -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -WindowStyle Hidden -PassThru
    if (-not (Wait-HttpReady 'http://127.0.0.1:9847/')) {
        Write-Diagnostics 'frontend readiness failure' $frontend $frontendOut $frontendErr
        throw 'Frontend did not become ready before the browser E2E deadline.'
    }
    $frontendReady = $true

    $testArguments = if ($Suite -eq 'LiveProvider') {
        @('app/tests/e2e/test_live_provider_flow.py', '-q')
    }
    else {
        @('app/tests/e2e', '-q')
    }
    & $python -m pytest @testArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Suite browser E2E suite failed with exit code $LASTEXITCODE."
    }
}
finally {
    if (-not $backendReady -or -not $frontendReady) {
        Write-Diagnostics 'browser E2E cleanup' $backend $backendOut $backendErr
        Write-Diagnostics 'browser E2E cleanup' $frontend $frontendOut $frontendErr
    }
    foreach ($process in @($backend, $frontend)) {
        if ($null -ne $process) {
            Stop-ProcessTree $process
            $process.Dispose()
        }
    }
}

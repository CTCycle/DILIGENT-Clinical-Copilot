[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Stop-Listeners {
    param([int[]]$Ports)
    $connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in $Ports }
    $processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($processId in $processIds) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

function Assert-PortFree {
    param([int[]]$Ports)
    foreach ($port in $Ports) {
        $listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -eq $port }
        if ($listeners) {
            $ids = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
            throw "Port $port is still in use by process id(s): $($ids -join ',')"
        }
    }
}

function Wait-Url {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 60
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Timed out waiting for $Url"
}

function Test-PythonModule {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ModuleName
    )
    & $PythonPath -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$serverDir = Join-Path $repoRoot "app\server"
$clientDir = Join-Path $repoRoot "app\client"
$backendPython = Join-Path $serverDir ".venv\Scripts\python.exe"
$uvExe = Join-Path $repoRoot "runtimes\uv\uv.exe"
$npmExe = Join-Path $repoRoot "runtimes\nodejs\npm.cmd"

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw "Missing backend interpreter: $backendPython"
}
if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "Missing uv runtime: $uvExe"
}
if (-not (Test-Path -LiteralPath $npmExe)) {
    throw "Missing npm runtime: $npmExe"
}

$backendUrl = "http://127.0.0.1:7690"
$frontendUrl = "http://127.0.0.1:9847"
$unitTestPath = Join-Path $repoRoot "app\tests\unit\test_model_config_persistence.py"
$e2eModelConfigPath = Join-Path $repoRoot "app\tests\e2e\test_model_config_api.py"
$e2eAppFlowPath = Join-Path $repoRoot "app\tests\e2e\test_app_flow.py"
if ($env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = $env:UV_CACHE_DIR.Trim()
}

Write-Host "[INFO] Stopping stale listeners on 7690/9847..."
Stop-Listeners -Ports @(7690, 9847)
Start-Sleep -Milliseconds 600
Assert-PortFree -Ports @(7690, 9847)

try {
    $logStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
    $tempRoot = $env:TEMP
    if (-not $tempRoot) { $tempRoot = $repoRoot }
    $runDbPath = Join-Path $tempRoot "codex-modelconfig-runner.$logStamp.db"
    $pytestCacheDir = Join-Path $tempRoot "codex-modelconfig-runner-pytest-cache.$logStamp"
    $env:DILIGENT_SQLITE_PATH = $runDbPath
    $env:PYTEST_ADDOPTS = "-o cache_dir=$pytestCacheDir"
    if (Test-Path -LiteralPath $runDbPath) {
        Remove-Item -LiteralPath $runDbPath -Force -ErrorAction SilentlyContinue
    }
    $backendOutLog = Join-Path $tempRoot "codex-modelconfig-runner-backend.$logStamp.out.log"
    $backendErrLog = Join-Path $tempRoot "codex-modelconfig-runner-backend.$logStamp.err.log"
    $frontendOutLog = Join-Path $tempRoot "codex-modelconfig-runner-frontend.$logStamp.out.log"
    $frontendErrLog = Join-Path $tempRoot "codex-modelconfig-runner-frontend.$logStamp.err.log"

    Write-Host "[INFO] Starting backend..."
    Start-Process -FilePath $backendPython `
        -ArgumentList "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7690" `
        -WorkingDirectory $serverDir `
        -RedirectStandardOutput $backendOutLog `
        -RedirectStandardError $backendErrLog `
        -WindowStyle Hidden

    Write-Host "[INFO] Starting frontend..."
    Start-Process -FilePath $npmExe `
        -ArgumentList "run", "start" `
        -WorkingDirectory $clientDir `
        -RedirectStandardOutput $frontendOutLog `
        -RedirectStandardError $frontendErrLog `
        -WindowStyle Hidden

    Write-Host "[INFO] Waiting for backend/frontend health..."
    Wait-Url -Url "$backendUrl/api/health" -TimeoutSeconds 90
    Wait-Url -Url $frontendUrl -TimeoutSeconds 90
    Start-Sleep -Milliseconds 750
    Wait-Url -Url "$backendUrl/api/model-config" -TimeoutSeconds 30

    Write-Host "[STEP] Running model-config unit tests..."
    if (Test-PythonModule -PythonPath $backendPython -ModuleName "pytest") {
        & $backendPython -m pytest $unitTestPath -q
    } else {
        & $uvExe run --python $backendPython --with pytest pytest $unitTestPath -q
    }
    if ($LASTEXITCODE -ne 0) { throw "Unit test command failed with exit code $LASTEXITCODE" }

    Write-Host "[STEP] Running model-config API + app-flow e2e slice..."
    $env:APP_TEST_FRONTEND_URL = $frontendUrl
    $env:APP_TEST_BACKEND_URL = $backendUrl
    if (
        (Test-PythonModule -PythonPath $backendPython -ModuleName "pytest") -and
        (Test-PythonModule -PythonPath $backendPython -ModuleName "pytest_playwright")
    ) {
        & $backendPython -m pytest `
            $e2eModelConfigPath $e2eAppFlowPath `
            -k "runtime_toggle_enables_save_and_submits_put or model_config or dili_run_burst_click_submits_single_job or dili_run_conflict_surfaces_clear_error_message" -q
    } else {
        & $uvExe run --python $backendPython --with pytest --with pytest-playwright `
            pytest $e2eModelConfigPath $e2eAppFlowPath `
            -k "runtime_toggle_enables_save_and_submits_put or model_config or dili_run_burst_click_submits_single_job or dili_run_conflict_surfaces_clear_error_message" -q
    }
    if ($LASTEXITCODE -ne 0) { throw "E2E test command failed with exit code $LASTEXITCODE" }

    Write-Host "[DONE] Model-config regression slice passed."
}
finally {
    Remove-Item Env:DILIGENT_SQLITE_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTEST_ADDOPTS -ErrorAction SilentlyContinue
    Write-Host "[INFO] Cleaning up listeners on 7690/9847..."
    Stop-Listeners -Ports @(7690, 9847)
}

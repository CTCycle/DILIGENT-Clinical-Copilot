[CmdletBinding()]
param(
    [ValidateSet('Full', 'LiveProvider')]
    [string]$Suite = 'Full'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$serverDir = Join-Path $repoRoot 'app/server'
$clientDir = Join-Path $repoRoot 'app/client'
$cacheRoot = if ([string]::IsNullOrWhiteSpace($env:DILIGENT_BROWSER_E2E_CACHE_ROOT)) {
    Join-Path $repoRoot 'runtimes/cache'
}
else {
    $env:DILIGENT_BROWSER_E2E_CACHE_ROOT
}
$pytestCacheDir = Join-Path $cacheRoot 'pytest'
$playwrightBrowsersPath = if ([string]::IsNullOrWhiteSpace($env:PLAYWRIGHT_BROWSERS_PATH)) {
    Join-Path $repoRoot 'runtimes/cache/playwright'
}
else {
    $env:PLAYWRIGHT_BROWSERS_PATH
}
$python = Join-Path $serverDir '.venv/Scripts/python.exe'
$logDir = Join-Path $pytestCacheDir 'browser-e2e-logs'
$backendOut = Join-Path $logDir 'backend.out.log'
$backendErr = Join-Path $logDir 'backend.err.log'
$frontendOut = Join-Path $logDir 'frontend.out.log'
$frontendErr = Join-Path $logDir 'frontend.err.log'
$ollamaOut = Join-Path $logDir 'ollama.out.log'
$ollamaErr = Join-Path $logDir 'ollama.err.log'
$junitPath = Join-Path $logDir "$($Suite.ToLowerInvariant())-junit.xml"
$backend = $null
$frontend = $null
$fakeOllama = $null
$backendReady = $false
$frontendReady = $false
$fakeOllamaReady = $false
$suiteExitCode = 1

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
$runRoot = Join-Path $pytestCacheDir "browser-e2e-$Suite"
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$databasePath = $env:DILIGENT_SQLITE_PATH
if ([string]::IsNullOrWhiteSpace($databasePath)) {
    $databasePath = Join-Path $runRoot 'diligent.sqlite3'
}
$databaseParent = Split-Path -Parent $databasePath
if ([string]::IsNullOrWhiteSpace($databaseParent)) {
    $databaseParent = $runRoot
}
New-Item -ItemType Directory -Path $databaseParent -Force | Out-Null
$env:DATABASE_BACKEND = 'sqlite'
$env:EMBEDDED_DATABASE = 'true'
$env:DATABASE_SQLITE_PATH = $databasePath
$env:DILIGENT_SQLITE_PATH = $databasePath
$env:DILIGENT_ACCESS_KEY_MATERIAL_FILE = Join-Path $runRoot 'access-key-material.json'
$env:APP_TEST_FRONTEND_URL = 'http://127.0.0.1:9847'
$env:APP_TEST_BACKEND_URL = 'http://127.0.0.1:7690'
$env:FASTAPI_HOST = '127.0.0.1'
$env:FASTAPI_PORT = '7690'
$env:UI_HOST = '127.0.0.1'
$env:UI_PORT = '9847'
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:PIP_CACHE_DIR = Join-Path $cacheRoot 'pip'
$env:NPM_CONFIG_CACHE = Join-Path $cacheRoot 'npm'
$env:PLAYWRIGHT_BROWSERS_PATH = $playwrightBrowsersPath
$env:HF_HOME = Join-Path $cacheRoot 'huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_ASSETS_CACHE = Join-Path $env:HF_HOME 'assets'
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:RUFF_CACHE_DIR = Join-Path $cacheRoot 'ruff'
$env:MYPY_CACHE_DIR = Join-Path $cacheRoot 'mypy'
$env:COVERAGE_FILE = Join-Path $cacheRoot 'coverage/.coverage'
$env:PYTHONPYCACHEPREFIX = Join-Path $cacheRoot 'python'
$env:CARGO_TARGET_DIR = Join-Path $cacheRoot 'cargo/target'
$env:DILIGENT_LIVE_PROVIDER_LOG_DIR = $logDir
$env:DILIGENT_PYTEST_CACHE_ROOT = $pytestCacheDir
$env:PYTEST_ADDOPTS = "--basetemp=`"$(Join-Path $pytestCacheDir 'basetemp')`" -o cache_dir=`"$pytestCacheDir`""
try {
    if ($Suite -eq 'Full') {
        $fakeOllamaPort = 11435
        $env:OLLAMA_URL = "http://127.0.0.1:$fakeOllamaPort"
        $seedScript = Join-Path $repoRoot 'app/tests/ci/seed_browser_regression_database.py'
        & $python $seedScript
        if ($LASTEXITCODE -ne 0) {
            throw "Browser regression database seed failed with exit code $LASTEXITCODE."
        }
        $fakeOllamaScript = Join-Path $repoRoot 'app/tests/ci/fake_ollama.py'
        $fakeOllamaArguments = @(
            "`"$fakeOllamaScript`"",
            '--port',
            "$fakeOllamaPort"
        )
        $fakeOllama = Start-Process -FilePath $python -ArgumentList $fakeOllamaArguments -WorkingDirectory $serverDir -RedirectStandardOutput $ollamaOut -RedirectStandardError $ollamaErr -WindowStyle Hidden -PassThru
        if (-not (Wait-HttpReady "http://127.0.0.1:$fakeOllamaPort/api/tags" -timeoutSeconds 30)) {
            Write-Diagnostics 'fake Ollama readiness failure' $fakeOllama $ollamaOut $ollamaErr
            throw 'Fake Ollama did not become ready before the browser E2E deadline.'
        }
        $fakeOllamaReady = $true
    }

    $backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '7690' -WorkingDirectory $serverDir -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -WindowStyle Hidden -PassThru
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
        @(
            (Join-Path $repoRoot 'app/tests/e2e/test_live_provider_flow.py'),
            '-q',
            '-rs',
            '--junitxml',
            $junitPath
        )
    }
    else {
        @(
            (Join-Path $repoRoot 'app/tests/e2e'),
            '-q',
            '-rs',
            '--ignore',
            (Join-Path $repoRoot 'app/tests/e2e/test_live_provider_flow.py'),
            '--ignore',
            (Join-Path $repoRoot 'app/tests/e2e/test_multilingual_embedding_runtime.py'),
            '--junitxml',
            $junitPath
        )
    }
    & $python -m pytest @testArguments
    $pytestExitCode = $LASTEXITCODE
    if ($pytestExitCode -ne 0) {
        throw "$Suite browser E2E suite failed with exit code $pytestExitCode."
    }
    if ($Suite -eq 'Full') {
        if (-not (Test-Path -LiteralPath $junitPath)) {
            throw "Mandatory browser E2E did not write its JUnit result: $junitPath"
        }
        [xml]$junit = Get-Content -LiteralPath $junitPath -Raw
        $suiteNodes = @($junit.SelectNodes('//testsuite'))
        $skippedCount = (
            $suiteNodes |
                ForEach-Object { [int]($_.GetAttribute('skipped')) } |
                Measure-Object -Sum
        ).Sum
        if ($skippedCount -ne 0) {
            throw "Mandatory browser E2E reported $skippedCount skipped test(s); optional integrations must be selected outside Suite Full."
        }
    }
    $suiteExitCode = 0
}
finally {
    if (-not $backendReady -or -not $frontendReady -or ($Suite -eq 'Full' -and -not $fakeOllamaReady)) {
        Write-Diagnostics 'browser E2E cleanup' $backend $backendOut $backendErr
        Write-Diagnostics 'browser E2E cleanup' $frontend $frontendOut $frontendErr
        Write-Diagnostics 'browser E2E cleanup' $fakeOllama $ollamaOut $ollamaErr
    }
    foreach ($process in @($backend, $frontend, $fakeOllama)) {
        if ($null -ne $process) {
            Stop-ProcessTree $process
            $process.Dispose()
        }
    }
}

exit $suiteExitCode

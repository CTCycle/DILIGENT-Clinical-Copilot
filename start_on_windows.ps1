[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$script:RepoRoot = $PSScriptRoot
$script:RuntimesDir = Join-Path $RepoRoot 'runtimes'
$script:PythonDir = Join-Path $RuntimesDir 'python'
$script:PythonExe = Join-Path $PythonDir 'python.exe'
$script:PythonPth = Join-Path $PythonDir 'python314._pth'
$script:UvDir = Join-Path $RuntimesDir 'uv'
$script:UvExe = Join-Path $UvDir 'uv.exe'
$script:NodeDir = Join-Path $RuntimesDir 'nodejs'
$script:NodeExe = Join-Path $NodeDir 'node.exe'
$script:NpmCmd = Join-Path $NodeDir 'npm.cmd'
$script:ServerDir = Join-Path $RepoRoot 'app/server'
$script:ClientDir = Join-Path $RepoRoot 'app/client'
$script:VenvDir = Join-Path $ServerDir '.venv'
$script:VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
$script:UvCacheDir = Join-Path $RuntimesDir '.uv-cache'
$script:EnvFile = Join-Path $RepoRoot 'settings/.env'
$script:EnvExample = Join-Path $RepoRoot 'settings/.env.example'
$script:PythonVersion = '3.14.2'
$script:NodeVersion = '22.12.0'

function Write-Step([string]$Message) {
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Info([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor DarkCyan
}

function Write-Fatal([string]$Message) {
    Write-Host "[FATAL] $Message" -ForegroundColor Red
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $RepoRoot
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($ArgumentList -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )
    $prevProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivePath) -Force | Out-Null
        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
        Invoke-WebRequest -Uri $Uri -OutFile $ArchivePath
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
        Remove-Item -LiteralPath $ArchivePath -Force
    }
    finally {
        $ProgressPreference = $prevProgress
    }
}

function Patch-PythonPth {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        (Get-Content -LiteralPath $Path) -replace '^#import site$', 'import site' | Set-Content -LiteralPath $Path
    }
}

function Invoke-PythonVersionCheck {
    param([Parameter(Mandatory = $true)][string]$PythonExecutable)
    & $PythonExecutable -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0) {
        throw "Python version check failed with exit code $LASTEXITCODE"
    }
}

function Find-UvExecutable {
    param([Parameter(Mandatory = $true)][string]$SearchRoot)
    $uv = Get-ChildItem -LiteralPath $SearchRoot -Recurse -Filter 'uv.exe' -File |
        Select-Object -First 1
    if ($null -eq $uv) {
        throw "uv.exe was not found under $SearchRoot"
    }
    $uv.FullName
}

function Invoke-HealthCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [ValidateRange(1, 600)][int]$Attempts = 60,
        [ValidateRange(1, 60)][int]$DelaySeconds = 1
    )
    $prevEA = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                    return $true
                }
            }
            catch {
            }
            if ($attempt -lt $Attempts) {
                Start-Sleep -Seconds $DelaySeconds
            }
        }
        return $false
    }
    finally {
        $ErrorActionPreference = $prevEA
    }
}

function Initialize-PortableRuntimes {
    Write-Step 'Ensuring portable Python, uv, and Node.js runtimes'
    New-Item -ItemType Directory -Path $PythonDir, $UvDir, $NodeDir -Force | Out-Null

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $pythonZipName = "python-$PythonVersion-embed-amd64.zip"
        $pythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$pythonZipName"
        Write-Info "Downloading $pythonUrl"
        Invoke-DownloadAndExtract `
            -Uri $pythonUrl `
            -ArchivePath (Join-Path $PythonDir $pythonZipName) `
            -DestinationPath $PythonDir
    }

    Patch-PythonPth -Path $PythonPth
    $pythonVersionFound = Invoke-PythonVersionCheck -PythonExecutable $PythonExe
    Write-Ok "Python ready: $pythonVersionFound"

    if (-not (Test-Path -LiteralPath $UvExe)) {
        $uvTarget = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
            'uv-aarch64-pc-windows-msvc.zip'
        }
        else {
            'uv-x86_64-pc-windows-msvc.zip'
        }
        $uvUrl = "https://github.com/astral-sh/uv/releases/latest/download/$uvTarget"
        Write-Info "Downloading $uvUrl"
        Invoke-DownloadAndExtract `
            -Uri $uvUrl `
            -ArchivePath (Join-Path $UvDir 'uv.zip') `
            -DestinationPath $UvDir
        $foundUv = Find-UvExecutable -SearchRoot $UvDir
        if ($foundUv -ne $UvExe) {
            Copy-Item -LiteralPath $foundUv -Destination $UvExe -Force
        }
    }
    Write-Ok (& $UvExe --version)

    if (-not (Test-Path -LiteralPath $NodeExe)) {
        $nodeZipName = "node-v$NodeVersion-win-x64.zip"
        $nodeUrl = "https://nodejs.org/dist/v$NodeVersion/$nodeZipName"
        Write-Info "Downloading $nodeUrl"
        Invoke-DownloadAndExtract `
            -Uri $nodeUrl `
            -ArchivePath (Join-Path $NodeDir $nodeZipName) `
            -DestinationPath $NodeDir

        $expandedNodeDir = Join-Path $NodeDir "node-v$NodeVersion-win-x64"
        if (Test-Path -LiteralPath (Join-Path $expandedNodeDir 'node.exe')) {
            Get-ChildItem -LiteralPath $expandedNodeDir -Force | ForEach-Object {
                Move-Item -LiteralPath $_.FullName -Destination $NodeDir -Force
            }
            Remove-Item -LiteralPath $expandedNodeDir -Recurse -Force
        }
    }

    if (-not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)) {
        throw "Portable Node.js or npm is missing under $NodeDir"
    }
    $env:PATH = "$NodeDir;$env:PATH"
    Write-Ok "Node.js ready: $(& $NodeExe --version)"
}

function Import-DotEnv {
    param([switch]$CreateIfMissing)

    if (-not (Test-Path -LiteralPath $EnvFile)) {
        if (-not $CreateIfMissing) {
            return
        }
        if (-not (Test-Path -LiteralPath $EnvExample)) {
            throw "Missing environment template: $EnvExample"
        }
        Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
        Write-Info "Created settings/.env from settings/.env.example"
    }

    foreach ($rawLine in Get-Content -LiteralPath $EnvFile) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#') -or $line.StartsWith(';')) {
            continue
        }
        $separator = $line.IndexOf('=')
        if ($separator -lt 1) {
            continue
        }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($key, $value, 'Process')
    }
}

function Set-LauncherEnvironment {
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    $env:UV_LINK_MODE = 'copy'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
}

function Install-ApplicationDependencies {
    Initialize-PortableRuntimes
    Import-DotEnv
    Set-LauncherEnvironment

    Write-Step 'Installing Python dependencies'
    $syncArguments = @('sync', '--python', $PythonExe)
    if ($env:OPTIONAL_DEPENDENCIES -eq 'true') {
        $syncArguments += '--all-extras'
    }
    try {
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArguments -WorkingDirectory $ServerDir
    }
    catch {
        Write-Info 'Recreating a virtual environment that may reference an older repository location'
        if (Test-Path -LiteralPath $VenvDir) {
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArguments -WorkingDirectory $ServerDir
    }

    Write-Step 'Installing frontend dependencies'
    $npmInstall = if (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) { 'ci' } else { 'install' }
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @($npmInstall) -WorkingDirectory $ClientDir

    Write-Step 'Building frontend'
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
    Write-Ok 'Dependencies and frontend build are ready'
}

function Get-ListeningProcessIds([int]$Port) {
    $pattern = ":$Port\s+.*LISTENING\s+(\d+)\s*$"
    foreach ($line in (& netstat.exe -ano -p tcp)) {
        if ($line -match $pattern) {
            [int]$Matches[1]
        }
    }
}

function Stop-PortListeners([int]$Port) {
    $processIds = @(Get-ListeningProcessIds -Port $Port | Sort-Object -Unique)
    foreach ($processId in $processIds) {
        Write-Info "Releasing port $Port from PID $processId"
        & taskkill.exe /PID $processId /F | Out-Null
    }

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        if (@(Get-ListeningProcessIds -Port $Port).Count -eq 0) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Port $Port is still occupied after 20 seconds"
}

function Start-Application {
    Import-DotEnv -CreateIfMissing
    Install-ApplicationDependencies
    Set-LauncherEnvironment

    $fastApiHost = if ($env:FASTAPI_HOST) { $env:FASTAPI_HOST } else { '127.0.0.1' }
    $fastApiPort = if ($env:FASTAPI_PORT) { [int]$env:FASTAPI_PORT } else { 8000 }
    $uiHost = if ($env:UI_HOST) { $env:UI_HOST } else { '127.0.0.1' }
    $uiPort = if ($env:UI_PORT) { [int]$env:UI_PORT } else { 7861 }
    $reload = $env:RELOAD -eq 'true'
    $backendVisible = $env:BACKEND_VISIBLE -eq 'true'

    Stop-PortListeners -Port $fastApiPort
    Stop-PortListeners -Port $uiPort

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Virtual environment Python is missing: $VenvPython"
    }

    $backendArguments = @(
        '-m', 'uvicorn', 'app:app', '--app-dir', (Join-Path $RepoRoot 'app'),
        '--host', $fastApiHost, '--port', [string]$fastApiPort, '--log-level', 'info'
    )
    if ($reload) {
        $backendArguments += '--reload'
    }

    Write-Step 'Launching backend'
    $backendProcess = $null
    if ($backendVisible) {
        $quotedArguments = $backendArguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
        $backendCommand = '"{0}" {1}' -f $VenvPython, ($quotedArguments -join ' ')
        Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', "start `"Backend`" cmd /c $backendCommand") -Wait
    }
    else {
        $backendProcess = Start-Process -FilePath $VenvPython -ArgumentList $backendArguments `
            -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
    }

    $healthUrl = "http://$fastApiHost`:$fastApiPort/api/health"
    Write-Info "Waiting up to 60 seconds for $healthUrl"
    $healthy = Invoke-HealthCheck -Url $healthUrl -Attempts 60 -DelaySeconds 1
    if (-not $healthy) {
        if ($backendProcess -and -not $backendProcess.HasExited) {
            Stop-Process -Id $backendProcess.Id -Force
        }
        throw "Backend did not become healthy at $healthUrl"
    }

    $backendPid = @(Get-ListeningProcessIds -Port $fastApiPort | Select-Object -First 1)
    Write-Step 'Launching frontend preview'
    $previewCommand = '"{0}" run preview -- --host "{1}" --port {2} --strictPort' -f $NpmCmd, $uiHost, $uiPort
    $frontendProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d', '/c', $previewCommand) `
        -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru

    $uiUrl = "http://$uiHost`:$uiPort"
    Start-Process $uiUrl

    Write-Host ''
    Write-Ok 'DILIGENT started successfully'
    Write-Host "Backend: $healthUrl (PID $($backendPid -join ','))"
    Write-Host "Frontend: $uiUrl (launcher PID $($frontendProcess.Id))"
}

function Install-OrUpdateApplication {
    Install-ApplicationDependencies
    if (Test-Path -LiteralPath $UvCacheDir) {
        Write-Step 'Pruning uv cache'
        Remove-Item -LiteralPath $UvCacheDir -Recurse -Force
    }
    Write-Ok 'Installation/update completed'
}

function Initialize-Database {
    Initialize-PortableRuntimes
    Set-LauncherEnvironment
    $databaseScript = Join-Path $RepoRoot 'app/scripts/initialize_database.py'
    if (-not (Test-Path -LiteralPath $databaseScript)) {
        throw "Database initializer is missing: $databaseScript"
    }
    Invoke-Checked -FilePath $UvExe -WorkingDirectory $RepoRoot -ArgumentList @(
        'run', '--project', 'app/server', '--python', $PythonExe, 'python',
        'app/scripts/initialize_database.py', '--drop-existing', '--seed-catalogs',
        '--force-reseed-catalogs'
    )
    Write-Ok 'Database initialization completed'
}

function Invoke-TestSuite {
    $testScript = Join-Path $RepoRoot 'app/tests/run_tests.bat'
    if (-not (Test-Path -LiteralPath $testScript)) {
        throw "Test runner is missing: $testScript"
    }
    & $testScript
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Test suite failed with exit code $exitCode"
    }
    Write-Ok 'Test suite completed'
}

function Remove-ApplicationLogs {
    $logDir = Join-Path $RepoRoot 'app/resources/logs'
    if (-not (Test-Path -LiteralPath $logDir)) {
        Write-Info "Log directory does not exist: $logDir"
        return
    }
    Get-ChildItem -LiteralPath $logDir -Filter '*.log' -File | Remove-Item -Force
    Write-Ok 'Application logs removed'
}

function Remove-PythonCaches {
    Get-ChildItem -LiteralPath $RepoRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object Name -eq '__pycache__' |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force -ErrorAction Continue
}

function Clear-ApplicationCache {
    Remove-PythonCaches
    if (Test-Path -LiteralPath $UvCacheDir) {
        Remove-Item -LiteralPath $UvCacheDir -Recurse -Force
    }
    Write-Ok 'Python and uv caches removed'
}

function Uninstall-Application {
    $targets = @(
        $RuntimesDir,
        $VenvDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist')
    )
    foreach ($target in $targets) {
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }

    $files = @(
        (Join-Path $ClientDir 'package-lock.json'),
        (Join-Path $ServerDir 'uv.lock'),
        (Join-Path $RepoRoot 'uv.lock')
    )
    foreach ($file in $files) {
        if (Test-Path -LiteralPath $file) {
            Remove-Item -LiteralPath $file -Force
        }
    }
    Remove-PythonCaches
    Write-Ok 'Application runtimes and generated dependencies removed; settings and user data were preserved'
}

function Wait-ForMenu {
    Write-Host ''
    Write-Host 'Press any key to return to menu...'
    [Console]::ReadKey($true) | Out-Null
}

while ($true) {
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        Clear-Host
    }
    Write-Host '========================================='
    Write-Host '    DILIGENT Clinical Copilot'
    Write-Host '========================================='
    Write-Host '1.  Launch application'
    Write-Host '2.  Install / update dependencies'
    Write-Host '3.  Initialize database'
    Write-Host '4.  Run test suite'
    Write-Host '5.  Remove logs'
    Write-Host '6.  Clear cache'
    Write-Host '7.  Uninstall application'
    Write-Host '8.  Exit'
    Write-Host '========================================='
    $rawSelection = Read-Host 'Select an option (1-8)'
    if ($null -eq $rawSelection) {
        exit 0
    }
    $selection = $rawSelection.Trim()

    try {
        switch -Regex ($selection) {
            '^1$' {
                Start-Application
                exit 0
            }
            '^2$' { Install-OrUpdateApplication }
            '^3$' { Initialize-Database }
            '^4$' { Invoke-TestSuite }
            '^5$' { Remove-ApplicationLogs }
            '^6$' { Clear-ApplicationCache }
            '^7$' { Uninstall-Application }
            '^8$' { exit 0 }
            default {
                Write-Host '[ERROR] Select a number from 1 through 8.' -ForegroundColor Red
                Wait-ForMenu
                continue
            }
        }
    }
    catch {
        Write-Fatal $_.Exception.Message
    }

    Wait-ForMenu
}

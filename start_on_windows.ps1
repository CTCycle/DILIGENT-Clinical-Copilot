[CmdletBinding()]
param(
    [ValidateSet('Launch', 'Install', 'InitializeDatabase', 'Test', 'Uninstall', 'BuildDesktopRelease', 'RemoveDesktopRelease')]
    [string]$Action,
    [ValidateSet('Standard', 'Development')]
    [string]$InstallationType,
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,
    [ValidateSet('Portable', 'Msi', 'All')]
    [string]$DesktopTarget = 'All',
    [switch]$OfflineWebView2,
    [switch]$AllDesktopReleases,
    [switch]$Force,
    [switch]$AllowDirtyTree
)

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
$script:NodeVersion = '22.13.0'
$script:DesktopDir = Join-Path $RepoRoot 'app/desktop'
$script:DesktopTauriDir = Join-Path $DesktopDir 'src-tauri'
$script:DesktopBuildDir = Join-Path $RepoRoot 'release/.staging'
$script:DesktopGeneratedDir = Join-Path $DesktopTauriDir 'generated'
$script:DesktopArtifactsDir = Join-Path $RepoRoot 'release'
$script:DesktopStageRoot = Join-Path $DesktopBuildDir 'staging'
$script:DesktopNpmCmd = Join-Path $NodeDir 'npm.cmd'
$script:DesktopTauriCli = Join-Path $DesktopDir 'node_modules/.bin/tauri.cmd'
$script:PyInstallerVersion = '6.21.0'

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

    $nodeNeedsInstall = -not (Test-Path -LiteralPath $NodeExe)
    if (-not $nodeNeedsInstall) {
        $installedNodeVersion = (& $NodeExe --version).Trim()
        $nodeNeedsInstall = $installedNodeVersion -ne "v$NodeVersion"
        if ($nodeNeedsInstall) {
            Write-Info "Replacing unsupported portable Node.js $installedNodeVersion with v$NodeVersion"
        }
    }

    if ($nodeNeedsInstall) {
        $nodeZipName = "node-v$NodeVersion-win-x64.zip"
        $nodeUrl = "https://nodejs.org/dist/v$NodeVersion/$nodeZipName"
        $nodeStageDir = Join-Path $RuntimesDir "nodejs-staging-$PID"
        Write-Info "Downloading $nodeUrl"
        try {
            Invoke-DownloadAndExtract `
                -Uri $nodeUrl `
                -ArchivePath (Join-Path $RuntimesDir $nodeZipName) `
                -DestinationPath $nodeStageDir

            $expandedNodeDir = Join-Path $nodeStageDir "node-v$NodeVersion-win-x64"
            if (-not (Test-Path -LiteralPath (Join-Path $expandedNodeDir 'node.exe'))) {
                throw "Downloaded Node.js archive did not contain the expected runtime"
            }

            Remove-Item -LiteralPath $NodeDir -Recurse -Force
            Move-Item -LiteralPath $expandedNodeDir -Destination $NodeDir
        }
        finally {
            if (Test-Path -LiteralPath $nodeStageDir) {
                Remove-Item -LiteralPath $nodeStageDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    if (-not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)) {
        throw "Portable Node.js or npm is missing under $NodeDir"
    }
    $env:PATH = "$NodeDir;$env:PATH"
    Write-Ok "Node.js ready: $(& $NodeExe --version)"
    Write-Ok 'Portable runtimes ready.'
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

    # Hosted Windows runners can reinsert Git/AWS/Python directories into
    # PATH between workflow steps. Keep the release venv's native Python DLLs
    # ahead of, and isolated from, any competing libffi/_ctypes pair before
    # starting the venv interpreter.
    $nativeNames = @('libffi-8.dll', 'python314.dll', 'python3.dll', '_ctypes.pyd')
    $cleanPath = foreach ($entry in ($env:PATH -split ';')) {
        if (-not $entry) {
            continue
        }
        $hasConflictingNativeRuntime = $false
        foreach ($nativeName in $nativeNames) {
            if (Test-Path -LiteralPath (Join-Path $entry $nativeName)) {
                $hasConflictingNativeRuntime = $true
                break
            }
        }
        if (-not $hasConflictingNativeRuntime) {
            $entry
        }
        else {
            Write-Info "Ignoring conflicting native-runtime PATH entry: $entry"
        }
    }
    $venvBin = Split-Path -Parent $VenvPython
    $env:PATH = "$venvBin;$($cleanPath -join ';')"
}

function Install-ApplicationDependencies {
    param(
        [bool]$BuildFrontend = $true,
        [ValidateSet('Standard', 'Development')]
        [string]$InstallationType = 'Standard',
        [switch]$PortableRuntimesReady
    )

    if (-not $PortableRuntimesReady) {
        Initialize-PortableRuntimes
    }
    Import-DotEnv
    Set-LauncherEnvironment

    Write-Step 'Installing Python dependencies'
    $syncArguments = @('sync', '--python', $PythonExe)
    if ($InstallationType -eq 'Development') {
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
    $nodeModules = Join-Path $ClientDir 'node_modules'
    $angularCli = Join-Path $nodeModules '@angular/cli/bin/ng.js'
    if (Test-Path -LiteralPath $angularCli) {
        Write-Info 'Reusing existing frontend dependencies'
    }
    elseif (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) {
        Write-Info 'Installing frontend dependencies from lockfile without lifecycle scripts'
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('ci', '--ignore-scripts', '--no-audit', '--no-fund') -WorkingDirectory $ClientDir
    }
    else {
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('install', '--ignore-scripts', '--no-audit', '--no-fund') -WorkingDirectory $ClientDir
    }

    if ($BuildFrontend) {
        Write-Step 'Building frontend'
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
        Write-Ok 'Dependencies and frontend build are ready'
    }
    else {
        Write-Info 'Skipping frontend build because ALWAYS_REBUILD=false'
        Write-Ok 'Dependencies are ready'
    }
}

function Test-DependenciesReady {
    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendInstallState = Join-Path $frontendModules '.package-lock.json'
    $frontendRunner = Join-Path $frontendModules '@angular/cli/bin/ng.js'
    $backendEntrypoint = Join-Path $ServerDir 'app.py'

    if (-not (Test-Path -LiteralPath $PythonExe) -or
        -not (Test-Path -LiteralPath $UvExe) -or
        -not (Test-Path -LiteralPath $NodeExe) -or
        -not (Test-Path -LiteralPath $NpmCmd) -or
        -not (Test-Path -LiteralPath $VenvPython) -or
        -not (Test-Path -LiteralPath $backendEntrypoint) -or
        -not (Test-Path -LiteralPath $frontendPackage) -or
        -not (Test-Path -LiteralPath $frontendLock) -or
        -not (Test-Path -LiteralPath $frontendInstallState) -or
        -not (Test-Path -LiteralPath $frontendRunner)) {
        return $false
    }

    & $PythonExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $UvExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $NodeExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $VenvPython -c 'import fastapi, uvicorn' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    return $true
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

function Convert-ToCommandLineArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    '"{0}"' -f ($Value -replace '"', '\\"')
}

function Get-BooleanEnvironmentValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Default
    )

    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    if ($value -notmatch '^(?i:true|false)$') {
        throw "$Name must be true or false when set"
    }
    return $value -ieq 'true'
}

function Start-Application {
    Import-DotEnv -CreateIfMissing
    $alwaysRebuild = Get-BooleanEnvironmentValue -Name 'ALWAYS_REBUILD' -Default $true
    Set-LauncherEnvironment
    if (-not (Test-DependenciesReady)) {
        Write-Step 'Required application environments are missing or unusable; installing dependencies'
        Install-ApplicationDependencies -BuildFrontend $alwaysRebuild -InstallationType 'Standard'
    }
    else {
        Write-Ok 'Application environments are ready; skipped dependency installation'
    }
    Set-LauncherEnvironment

    $fastApiHost = if ($env:FASTAPI_HOST) { $env:FASTAPI_HOST } else { '127.0.0.1' }
    $fastApiPort = if ($env:FASTAPI_PORT) { [int]$env:FASTAPI_PORT } else { 8000 }
    $uiHost = if ($env:UI_HOST) { $env:UI_HOST } else { '127.0.0.1' }
    $uiPort = if ($env:UI_PORT) { [int]$env:UI_PORT } else { 7861 }
    $reload = $env:RELOAD -eq 'true'
    $backendLogsVisible = Get-BooleanEnvironmentValue -Name 'BACKEND_LOGS_VISIBLE' -Default $true

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

    $backendCommandParts = @($VenvPython) + $backendArguments |
        ForEach-Object { Convert-ToCommandLineArgument -Value ([string]$_) }
    $backendCommand = '"{0}"' -f ($backendCommandParts -join ' ')

    Write-Step 'Launching backend'
    $backendProcess = $null
    if ($backendLogsVisible) {
        $backendProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d', '/k', $backendCommand) `
            -WorkingDirectory $RepoRoot -WindowStyle Normal -PassThru
    }
    else {
        $backendProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d', '/c', $backendCommand) `
            -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
    }

    $healthUrl = "http://$fastApiHost`:$fastApiPort/api/health"
    Write-Info "Waiting up to 60 seconds for $healthUrl"
    $healthy = Invoke-HealthCheck -Url $healthUrl -Attempts 60 -DelaySeconds 1
    if (-not $healthy) {
        if ($backendProcess -and -not $backendProcess.HasExited) {
            & taskkill.exe /PID $backendProcess.Id /T /F | Out-Null
        }
        throw "Backend did not become healthy at $healthUrl"
    }

    $backendPid = @(Get-ListeningProcessIds -Port $fastApiPort | Select-Object -First 1)
    Write-Step 'Launching frontend preview'
    $previewCommand = '"{0}" run preview -- --host "{1}" --port {2} --strictPort' -f $NpmCmd, $uiHost, $uiPort
    $previewCommand = '"{0}"' -f $previewCommand
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
    $selectedInstallationType = $InstallationType
    $portableRuntimesReady = $false
    if (-not $selectedInstallationType) {
        Initialize-PortableRuntimes
        $portableRuntimesReady = $true
        $selectedInstallationType = Read-InstallationType
    }
    Install-ApplicationDependencies `
        -InstallationType $selectedInstallationType `
        -PortableRuntimesReady:$portableRuntimesReady
    if (Test-Path -LiteralPath $UvCacheDir) {
        Write-Step 'Pruning uv cache'
        Remove-Item -LiteralPath $UvCacheDir -Recurse -Force
    }
    Write-Ok 'Installation/update completed'
}

function Initialize-Database {
    Initialize-PortableRuntimes
    Import-DotEnv -CreateIfMissing
    Set-LauncherEnvironment
    $databaseScript = Join-Path $RepoRoot 'app/scripts/initialize_database.py'
    if (-not (Test-Path -LiteralPath $databaseScript)) {
        throw "Database initializer is missing: $databaseScript"
    }
    Invoke-Checked -FilePath $UvExe -WorkingDirectory $RepoRoot -ArgumentList @(
        'run', '--project', 'app/server', '--python', $PythonExe, 'python',
        'app/scripts/initialize_database.py', '--seed-catalogs'
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

function Read-InstallationType {
    Write-Host '  [1] Development - include Ruff, Pyright, and pytest'
    Write-Host '  [2] Standard    - install runtime dependencies only'
    $selection = (Read-Host '  Select installation profile [1-2]').Trim()
    switch ($selection) {
        '1' { return 'Development' }
        '2' { return 'Standard' }
        default { throw 'Invalid installation type. Enter 1 for Development or 2 for Standard.' }
    }
}

function Assert-DesktopParameterContract {
    if ($OfflineWebView2 -and $Action -ne 'BuildDesktopRelease') {
        throw '-OfflineWebView2 is valid only with BuildDesktopRelease'
    }
    if ($OfflineWebView2 -and $DesktopTarget -eq 'Portable') {
        throw '-OfflineWebView2 requires an MSI target'
    }
    if ($AllDesktopReleases -and $Action -ne 'RemoveDesktopRelease') {
        throw '-AllDesktopReleases is valid only with RemoveDesktopRelease'
    }
    if ($Action -eq 'RemoveDesktopRelease' -and -not $AllDesktopReleases -and -not $Version) {
        throw 'RemoveDesktopRelease requires -Version or -AllDesktopReleases'
    }
}

function Resolve-DesktopReleaseVersion {
    if ($Version) {
        return $Version
    }
    $packagePath = Join-Path $ClientDir 'package.json'
    $defaultVersion = ((Get-Content -LiteralPath $packagePath -Raw | ConvertFrom-Json).version)
    $entered = Read-Host "Release version [$defaultVersion]"
    $resolved = if ([string]::IsNullOrWhiteSpace($entered)) { $defaultVersion } else { $entered.Trim() }
    if ($resolved -notmatch '^\d+\.\d+\.\d+$') {
        throw 'Release version must use major.minor.patch format'
    }
    return $resolved
}

function Get-RepositoryCommit {
    $commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $commit) { throw 'Unable to resolve the repository commit SHA' }
    return $commit
}

function Assert-CleanReleaseTree {
    $status = @(git -C $RepoRoot status --porcelain --untracked-files=all)
    if ($status.Count -gt 0 -and -not $AllowDirtyTree) {
        throw 'Desktop release builds require a clean worktree. Supply -AllowDirtyTree to record a dirty build explicitly.'
    }
    return ($status.Count -gt 0)
}

function Assert-DesktopBuildHost {
    if ($env:PROCESSOR_ARCHITECTURE -notin @('AMD64', 'x86')) {
        throw 'Desktop releases require a Windows x64 build host'
    }
    foreach ($tool in @('cargo', 'rustc')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Required desktop build tool is missing: $tool"
        }
    }
    if (-not (Get-Command msbuild -ErrorAction SilentlyContinue) -and -not $env:VisualStudioVersion) {
        Write-Info 'MSBuild was not found on PATH; Tauri will use the configured Visual Studio toolchain if available.'
    }
}

function Initialize-DesktopBuildDependencies {
    Install-ApplicationDependencies -BuildFrontend $true
    Write-Step "Installing PyInstaller $PyInstallerVersion in the build environment"
    Invoke-Checked -FilePath $UvExe -WorkingDirectory $ServerDir -ArgumentList @(
        'pip', 'install', '--python', $VenvPython, "pyinstaller==$PyInstallerVersion"
    )
    if (-not (Test-Path -LiteralPath $DesktopTauriCli)) {
        Invoke-Checked -FilePath $DesktopNpmCmd -WorkingDirectory $DesktopDir -ArgumentList @(
            'ci', '--ignore-scripts', '--no-audit', '--no-fund'
        )
    }
}

function Build-DesktopFrontend {
    Invoke-Checked -FilePath $DesktopNpmCmd -WorkingDirectory $ClientDir -ArgumentList @('run', 'build')
    if (-not (Test-Path -LiteralPath (Join-Path $ClientDir 'dist/browser/index.html'))) {
        throw 'Angular production output was not generated'
    }
}

function Build-DesktopBackend {
    param([Parameter(Mandatory = $true)][string]$StageRoot)
    $distPath = Join-Path $StageRoot 'backend'
    $workPath = Join-Path $DesktopBuildDir 'pyinstaller-work'
    New-Item -ItemType Directory -Path $distPath, $workPath -Force | Out-Null
    Invoke-Checked -FilePath $VenvPython -WorkingDirectory $RepoRoot -ArgumentList @(
        (Join-Path $DesktopDir 'build/run_pyinstaller.py'), '--noconfirm', '--clean', '--distpath', $distPath,
        '--workpath', $workPath,
        (Join-Path $DesktopDir 'build/diligent_backend.spec')
    )
    $backendRoot = Join-Path $distPath 'DILIGENTBackend'
    $backendExecutable = Join-Path $backendRoot 'DILIGENTBackend.exe'
    if (-not (Test-Path -LiteralPath $backendExecutable)) { throw "Frozen backend was not created: $backendExecutable" }
    Copy-Item -Path (Join-Path $backendRoot '*') -Destination $distPath -Recurse -Force
    Remove-Item -LiteralPath $backendRoot -Recurse -Force
}

function Test-FrozenBackend {
    param([Parameter(Mandatory = $true)][string]$StageRoot, [Parameter(Mandatory = $true)][string]$Version)
    $backendRoot = Join-Path $StageRoot 'runtime-stage/backend'
    $backend = Join-Path $backendRoot 'DILIGENTBackend.exe'
    $dataRoot = Join-Path $StageRoot 'frozen-data'
    $readyFile = Join-Path $dataRoot 'state/ready.json'
    New-Item -ItemType Directory -Path (Split-Path -Parent $readyFile) -Force | Out-Null
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $backend
    $psi.Arguments = "--ready-file `"$readyFile`" --host 127.0.0.1"
    $psi.WorkingDirectory = $backendRoot
    $psi.UseShellExecute = $false
    foreach ($pair in @{
        DILIGENT_DESKTOP='true'; DILIGENT_RELEASE_VERSION=$Version; DILIGENT_RUNTIME_ROOT=(Join-Path $StageRoot 'runtime-stage')
        DILIGENT_DATA_ROOT=$dataRoot; DILIGENT_SQLITE_PATH=(Join-Path $dataRoot 'resources/database.db')
        DILIGENT_ACCESS_KEY_MATERIAL_FILE=(Join-Path $dataRoot 'resources/access-key-material.json'); RELOAD='false'
    }.GetEnumerator()) { $psi.Environment[$pair.Key] = [string]$pair.Value }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $psi
    try {
        if (-not $process.Start()) { throw 'Frozen backend did not start' }
        $ready = $null
        for ($attempt = 0; $attempt -lt 120 -and $null -eq $ready; $attempt++) {
            if (Test-Path -LiteralPath $readyFile) { $ready = Get-Content -LiteralPath $readyFile -Raw | ConvertFrom-Json }
            elseif ($process.HasExited) { throw 'Frozen backend exited before its ready file appeared' }
            if ($null -eq $ready) { Start-Sleep -Milliseconds 500 }
        }
        if ($null -eq $ready) { throw 'Frozen backend ready-file timeout' }
        $baseUrl = "http://127.0.0.1:$($ready.port)"
        if (-not (Invoke-HealthCheck -Url "$baseUrl/api/health" -Attempts 60 -DelaySeconds 1)) { throw 'Frozen backend health check failed' }
        if ((Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/" -TimeoutSec 5).StatusCode -ne 200) { throw 'Frozen backend did not serve Angular index' }
        if ((Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/clinical-sessions" -TimeoutSec 5).StatusCode -ne 200) { throw 'Frozen backend SPA fallback failed' }
        Write-Ok 'Frozen backend smoke test passed'
    }
    finally {
        if ($process -and -not $process.HasExited) { $process.Kill(); $process.WaitForExit(10000) | Out-Null }
        Remove-Item -LiteralPath $dataRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function New-DesktopRuntimeArchive {
    param([Parameter(Mandatory = $true)][string]$StageRoot, [Parameter(Mandatory = $true)][string]$Version, [Parameter(Mandatory = $true)][bool]$DirtyTree)
    $payloadRoot = Join-Path $StageRoot 'runtime-stage'
    New-Item -ItemType Directory -Path $payloadRoot -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $StageRoot 'backend') -Destination (Join-Path $payloadRoot 'backend') -Recurse
    foreach ($relative in @('app/client/dist/browser', 'app/resources/catalogs', 'settings')) {
        New-Item -ItemType Directory -Path (Join-Path $payloadRoot $relative) -Force | Out-Null
    }
    Copy-Item -Path (Join-Path $ClientDir 'dist/browser/*') -Destination (Join-Path $payloadRoot 'app/client/dist/browser') -Recurse -Force
    Copy-Item -Path (Join-Path $RepoRoot 'app/resources/catalogs/*') -Destination (Join-Path $payloadRoot 'app/resources/catalogs') -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'settings/.env.example') -Destination (Join-Path $payloadRoot 'settings/.env.example')
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'settings/configurations.json') -Destination (Join-Path $payloadRoot 'settings/configurations.json')
    $runtimeOutput = Join-Path $StageRoot 'runtime/diligent-runtime.zip'
    $manifestOutput = Join-Path $StageRoot 'runtime/runtime-manifest.json'
    New-Item -ItemType Directory -Path (Split-Path -Parent $runtimeOutput) -Force | Out-Null
    $clientVersion = ((Get-Content -LiteralPath (Join-Path $ClientDir 'package.json') -Raw | ConvertFrom-Json).version)
    Invoke-Checked -FilePath $VenvPython -WorkingDirectory $RepoRoot -ArgumentList @(
        (Join-Path $DesktopDir 'build/create_runtime_bundle.py'), '--staging', $payloadRoot, '--version', $Version,
        '--output', $runtimeOutput, '--manifest', $manifestOutput, '--python-version', (& $VenvPython --version),
        '--pyinstaller-version', $PyInstallerVersion, '--tauri-version', '2.11.5', '--frontend-package-version', $clientVersion,
        '--source-commit-sha', (Get-RepositoryCommit), '--dirty-tree', ([string]$DirtyTree)
    )
    Copy-Item -LiteralPath $runtimeOutput -Destination (Join-Path $DesktopGeneratedDir 'diligent-runtime.zip') -Force
    $digest = (Get-FileHash -LiteralPath $runtimeOutput -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath (Join-Path $DesktopGeneratedDir 'diligent-runtime.sha256') -Value $digest -Encoding ascii
}

function New-TauriReleaseConfiguration {
    param([Parameter(Mandatory = $true)][string]$Version)
    $configuration = Get-Content -LiteralPath (Join-Path $DesktopTauriDir 'tauri.conf.json') -Raw | ConvertFrom-Json
    $configuration.version = $Version
    $configuration.bundle | Add-Member -MemberType NoteProperty -Name 'icon' -Value @(
        (Join-Path $DesktopTauriDir 'icons/icon.ico'),
        (Join-Path $DesktopTauriDir 'icons/32x32.png'),
        (Join-Path $DesktopTauriDir 'icons/128x128.png'),
        (Join-Path $DesktopTauriDir 'icons/128x128@2x.png')
    ) -Force
    $configuration.bundle.windows.webviewInstallMode.type = if ($OfflineWebView2) { 'offlineInstaller' } else { 'embedBootstrapper' }
    $path = Join-Path $DesktopBuildDir "tauri-$Version-$PID.json"
    $configuration | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $path -Encoding utf8
    return $path
}

function Build-TauriApplication {
    param([Parameter(Mandatory = $true)][string]$ConfigurationPath, [Parameter(Mandatory = $true)][string]$Version)
    if ([string]::IsNullOrWhiteSpace($env:CARGO_BUILD_JOBS)) {
        $env:CARGO_BUILD_JOBS = '1'
        Write-Info 'Using one Cargo build job to keep release compilation within host memory limits.'
    }
    Invoke-Checked -FilePath $DesktopTauriCli -WorkingDirectory $DesktopDir -ArgumentList @('build', '--config', $ConfigurationPath, '--no-bundle')
    if ($DesktopTarget -in @('Msi', 'All')) {
        Invoke-Checked -FilePath $DesktopTauriCli -WorkingDirectory $DesktopDir -ArgumentList @('bundle', '--config', $ConfigurationPath, '--bundles', 'msi')
    }
}

function Test-PortableDesktopArtifact {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -lt 1MB) { throw "Portable desktop artifact is missing or unexpectedly small: $Path" }
}

function Test-MsiMetadata {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -lt 1KB) { throw "MSI artifact is missing or unexpectedly small: $Path" }
}

function Publish-DesktopArtifacts {
    param([Parameter(Mandatory = $true)][string]$Version)
    New-Item -ItemType Directory -Path $DesktopArtifactsDir -Force | Out-Null
    $releasePrefix = "DILIGENT-v$Version-windows-x64"
    $portable = Join-Path $DesktopArtifactsDir "$releasePrefix-portable.exe"
    $msi = Join-Path $DesktopArtifactsDir "$releasePrefix.msi"
    $rawExe = Get-ChildItem -LiteralPath (Join-Path $DesktopTauriDir 'target/release') -Filter '*.exe' -File -ErrorAction SilentlyContinue | Where-Object Name -notmatch 'uninstall' | Select-Object -First 1
    if ($DesktopTarget -in @('Portable', 'All')) {
        if ($null -eq $rawExe) { throw 'Tauri raw executable was not found' }
        if ((Test-Path -LiteralPath $portable) -and -not $Force) { throw "Release already exists: $portable (use -Force to replace it)" }
        Copy-Item -LiteralPath $rawExe.FullName -Destination $portable -Force
        Test-PortableDesktopArtifact -Path $portable
    }
    if ($DesktopTarget -in @('Msi', 'All')) {
        $builtMsi = Get-ChildItem -LiteralPath (Join-Path $DesktopTauriDir 'target/release/bundle/msi') -Filter '*.msi' -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $builtMsi) { throw 'Tauri MSI artifact was not found' }
        if ((Test-Path -LiteralPath $msi) -and -not $Force) { throw "Release already exists: $msi (use -Force to replace it)" }
        Copy-Item -LiteralPath $builtMsi.FullName -Destination $msi -Force
        Test-MsiMetadata -Path $msi
    }
    Write-DesktopChecksums -Version $Version
}

function Write-DesktopChecksums {
    param([Parameter(Mandatory = $true)][string]$Version)
    $prefix = "DILIGENT-v$Version-windows-x64"
    $checksumPath = Join-Path $DesktopArtifactsDir "$prefix.sha256"
    $lines = @()
    foreach ($artifact in @("$prefix-portable.exe", "$prefix.msi")) {
        $path = Join-Path $DesktopArtifactsDir $artifact
        if (Test-Path -LiteralPath $path) { $lines += "SHA256  $artifact"; $lines += ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() + "  $artifact") }
    }
    Set-Content -LiteralPath $checksumPath -Value $lines -Encoding ascii
}

function Build-DesktopRelease {
    Assert-DesktopParameterContract
    $resolvedVersion = Resolve-DesktopReleaseVersion
    $dirtyTree = Assert-CleanReleaseTree
    Assert-DesktopBuildHost
    $stageRoot = Join-Path $DesktopStageRoot "$resolvedVersion/$PID"
    New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
    try {
        Initialize-DesktopBuildDependencies
        Build-DesktopFrontend
        Build-DesktopBackend -StageRoot $stageRoot
        New-DesktopRuntimeArchive -StageRoot $stageRoot -Version $resolvedVersion -DirtyTree $dirtyTree
        Test-FrozenBackend -StageRoot $stageRoot -Version $resolvedVersion
        $configuration = New-TauriReleaseConfiguration -Version $resolvedVersion
        Build-TauriApplication -ConfigurationPath $configuration -Version $resolvedVersion
        Publish-DesktopArtifacts -Version $resolvedVersion
        Write-Ok "Desktop release $resolvedVersion published under $DesktopArtifactsDir"
    }
    finally {
        if (Test-Path -LiteralPath $stageRoot) { Remove-Item -LiteralPath $stageRoot -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

function Get-DesktopReleaseVersions {
    if (-not (Test-Path -LiteralPath $DesktopArtifactsDir)) { return @() }
    Get-ChildItem -LiteralPath $DesktopArtifactsDir -File | ForEach-Object {
        if ($_.Name -match '^DILIGENT-v(\d+\.\d+\.\d+)-windows-x64') { $Matches[1] }
    } | Sort-Object -Unique
}

function Remove-DesktopRelease {
    Assert-DesktopParameterContract
    if ($AllDesktopReleases) {
        if (Test-Path -LiteralPath $DesktopArtifactsDir) { Get-ChildItem -LiteralPath $DesktopArtifactsDir -File | Where-Object Name -match '^DILIGENT-v\d+\.\d+\.\d+-windows-x64' | Remove-Item -Force }
    }
    else {
        foreach ($suffix in @('-portable.exe', '.msi', '.sha256')) {
            $target = Join-Path $DesktopArtifactsDir "DILIGENT-v$Version-windows-x64$suffix"
            if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }
        }
    }
    foreach ($target in @($DesktopBuildDir, $DesktopGeneratedDir, (Join-Path $DesktopTauriDir 'target'), (Join-Path $DesktopDir 'node_modules'))) {
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    Write-Ok 'Desktop release artifacts and generated desktop build state removed; installed applications and user data were preserved'
}

$script:MenuContentWidth = 72
$script:MenuFramePadding = 4

function Write-MenuRule {
    $rule = '-' * ($script:MenuContentWidth + $script:MenuFramePadding)
    Write-Host ('  +{0}+' -f $rule) -ForegroundColor DarkCyan
}

function Write-MenuLine {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )

    $displayText = $Text
    if ($displayText.Length -gt $script:MenuContentWidth) {
        $displayText = $displayText.Substring(0, $script:MenuContentWidth - 3) + '...'
    }
    Write-Host ('  |  ' + $displayText.PadRight($script:MenuContentWidth) + '  |') -ForegroundColor $Color
}

function Write-MenuOption {
    param(
        [Parameter(Mandatory = $true)][string]$Number,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $content = '{0}  {1,-31} {2}' -f $Number, $Label, $Description
    Write-MenuLine -Text $content
}

function Show-MainMenu {
    Write-Host ''
    Write-MenuRule
    Write-MenuLine -Text 'DILIGENT  /  CLINICAL COPILOT' -Color Cyan
    Write-MenuLine -Text 'Local development and maintenance console' -Color DarkGray
    Write-MenuRule
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text 'APPLICATION' -Color Yellow
    Write-MenuOption -Number '1.' -Label 'Launch application' -Description 'Start local services'
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text 'MAINTENANCE' -Color Yellow
    Write-MenuOption -Number '2.' -Label 'Install / update dependencies' -Description 'Sync runtimes + packages'
    Write-MenuOption -Number '3.' -Label 'Initialize database' -Description 'Prepare local data store'
    Write-MenuOption -Number '4.' -Label 'Run test suite' -Description 'Execute project checks'
    Write-MenuOption -Number '5.' -Label 'Remove logs' -Description 'Delete application logs'
    Write-MenuOption -Number '6.' -Label 'Clear cache' -Description 'Remove temporary caches'
    Write-MenuOption -Number '7.' -Label 'Uninstall application' -Description 'Remove generated files'
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text 'DESKTOP RELEASE' -Color Yellow
    Write-MenuOption -Number '8.' -Label 'Build desktop release' -Description 'Create portable / MSI packages'
    Write-MenuOption -Number '9.' -Label 'Remove desktop release artifacts' -Description 'Delete generated release state'
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text '10. Exit'
    Write-MenuRule
}

function Wait-ForMenu {
    Write-Host ''
    Write-Host 'Press any key to return to menu...'
    [Console]::ReadKey($true) | Out-Null
}

if ($Action) {
    switch ($Action) {
        'Launch' { Start-Application }
        'Install' { Install-OrUpdateApplication }
        'InitializeDatabase' { Initialize-Database }
        'Test' { Invoke-TestSuite }
        'Uninstall' { Uninstall-Application }
        'BuildDesktopRelease' { Build-DesktopRelease }
        'RemoveDesktopRelease' { Remove-DesktopRelease }
    }
    exit 0
}

while ($true) {
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        Clear-Host
    }
    Show-MainMenu
    $rawSelection = Read-Host 'Select an option (1-10)'
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
            '^8$' { Build-DesktopRelease; exit 0 }
            '^9$' { Remove-DesktopRelease }
            '^10$' { exit 0 }
            default {
                Write-Host '[ERROR] Select a number from 1 through 10.' -ForegroundColor Red
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

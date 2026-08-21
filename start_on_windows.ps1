# ============================================================
# Launcher parameters and repository paths
# ============================================================
[CmdletBinding()]
param(
    [ValidateSet('Launch', 'Install', 'RebuildFrontend', 'InitializeDatabase', 'Test', 'Uninstall', 'Update', 'CheckForUpdates', 'RemoveAllData', 'BuildDesktopRelease', 'RemoveDesktopRelease')]
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
$script:TestsDir = Join-Path $RepoRoot 'app/tests'
$script:VenvDir = Join-Path $ServerDir '.venv'
$script:VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
$script:RuntimeCacheDir = Join-Path $RuntimesDir 'cache'
$script:TestCacheDir = Join-Path $TestsDir 'cache'
$script:LegacyCacheDir = Join-Path $RepoRoot 'assets/cache'
$script:AngularCacheDir = Join-Path $TestCacheDir 'angular'
$script:CoverageDir = Join-Path $TestCacheDir 'coverage'
$script:CargoTargetDir = Join-Path $RuntimeCacheDir 'cargo'
$script:MypyCacheDir = Join-Path $TestCacheDir 'mypy'
$script:NpmCacheDir = Join-Path $RuntimeCacheDir 'npm'
$script:PipCacheDir = Join-Path $RuntimeCacheDir 'pip'
$script:PlaywrightCacheDir = Join-Path $RuntimeCacheDir 'playwright'
$script:PytestCacheDir = Join-Path $TestCacheDir 'pytest'
$script:PythonBytecodeCacheDir = Join-Path $RuntimeCacheDir 'python'
$script:RuffCacheDir = Join-Path $TestCacheDir 'ruff'
$script:UvCacheDir = Join-Path $RuntimeCacheDir 'uv'
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

# ============================================================
# Shared output and process helpers
# ============================================================
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

# ============================================================
# Portable runtimes and launcher environment
# ============================================================
function Initialize-PortableNodeRuntime {
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null

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
}

function Initialize-PortableRuntimes {
    Write-Step 'Ensuring portable Python, uv, and Node.js runtimes'
    New-Item -ItemType Directory -Path $PythonDir, $UvDir -Force | Out-Null

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

    Initialize-PortableNodeRuntime
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
    New-Item -ItemType Directory -Path @(
        $RuntimeCacheDir,
        $TestCacheDir,
        $AngularCacheDir,
        $CoverageDir,
        $CargoTargetDir,
        $MypyCacheDir,
        $NpmCacheDir,
        $PipCacheDir,
        $PlaywrightCacheDir,
        $PytestCacheDir,
        $PythonBytecodeCacheDir,
        $RuffCacheDir,
        $UvCacheDir
    ) -Force | Out-Null
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    $env:UV_LINK_MODE = 'copy'
    $env:PIP_CACHE_DIR = $PipCacheDir
    $env:NPM_CONFIG_CACHE = $NpmCacheDir
    $env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightCacheDir
    $env:RUFF_CACHE_DIR = $RuffCacheDir
    $env:MYPY_CACHE_DIR = $MypyCacheDir
    $env:COVERAGE_FILE = Join-Path $CoverageDir '.coverage'
    $env:PYTHONPYCACHEPREFIX = $PythonBytecodeCacheDir
    $env:CARGO_TARGET_DIR = $CargoTargetDir
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
        $hasConflictingNativeRuntime = (
            $entry -match '(?i)[\\/]Git[\\/]usr[\\/]bin(?:[\\/]|$)' -or
            $entry -match '(?i)(?:^|[\\/])mingw(?:64)?[\\/]bin(?:[\\/]|$)' -or
            $entry -match '(?i)[\\/]Amazon[\\/]AWSCLIV2(?:[\\/]|$)' -or
            $entry -match '(?i)[\\/]Python(?:[0-9.]*)?(?:[\\/]|$)'
        )
        if (-not $hasConflictingNativeRuntime) {
            foreach ($nativeName in $nativeNames) {
                if (Test-Path -LiteralPath (Join-Path $entry $nativeName)) {
                    $hasConflictingNativeRuntime = $true
                    break
                }
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
        [bool]$BuildFrontend = $false,
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

    Install-FrontendDependencies

    if ($BuildFrontend) {
        Build-Frontend
        Write-Ok 'Dependencies and frontend build are ready'
    }
    else {
        Write-Info 'Skipping frontend build; use the frontend rebuild option to rebuild the frontend'
        Write-Ok 'Dependencies are ready'
    }
}

function Install-FrontendDependencies {
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
}

function Build-Frontend {
    Write-Step 'Building frontend'
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
    if (-not (Test-Path -LiteralPath (Join-Path $ClientDir 'dist/browser/index.html') -PathType Leaf)) {
        throw 'Angular production output was not generated'
    }
}

# ============================================================
# Application lifecycle
# ============================================================
function Rebuild-Frontend {
    Initialize-PortableNodeRuntime
    Import-DotEnv
    Install-FrontendDependencies
    Build-Frontend
    Write-Ok 'Frontend rebuild completed'
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
    Set-LauncherEnvironment
    $frontendIndex = Join-Path $ClientDir 'dist/browser/index.html'
    $dependenciesReady = Test-DependenciesReady
    $frontendBuildReady = Test-Path -LiteralPath $frontendIndex -PathType Leaf
    if (-not $dependenciesReady -or -not $frontendBuildReady) {
        Write-Step 'Required application environments or frontend build are missing or unusable; recovering dependencies and frontend build'
        Install-ApplicationDependencies -BuildFrontend $true -InstallationType 'Standard'
    }
    else {
        Write-Ok 'Application environments are ready; skipped dependency installation'
    }
    if (-not (Test-Path -LiteralPath $frontendIndex)) {
        throw 'Frontend build output is still missing after recovery. Select option 2 to retry dependency installation and the frontend build, or use the frontend rebuild option when dependencies are ready.'
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

# ============================================================
# Source control and update status
# ============================================================
function Get-GitRevision {
    param(
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $revision = & git.exe -C $RepoRoot @ArgumentList 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $firstRevision = $revision | Select-Object -First 1
    if ($null -eq $firstRevision) {
        return $null
    }
    $value = ([string]$firstRevision).Trim()
    if ($value) {
        return $value
    }
    return $null
}

function Check-ForUpdates {
    Write-Step 'Checking origin/main for updates'
    $remoteResult = @(& git.exe -C $RepoRoot ls-remote origin refs/heads/main 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to check origin/main: $($remoteResult -join ' ')"
    }

    $remoteLine = $remoteResult |
        Where-Object { $_ -match '^[0-9a-f]{40}\s+refs/heads/main$' } |
        Select-Object -First 1
    if (-not $remoteLine) {
        throw 'Remote origin does not expose a main branch.'
    }
    $remoteRevision = ($remoteLine -split '\s+')[0]
    $localMainRevision = Get-GitRevision -ArgumentList @('rev-parse', '--verify', 'refs/remotes/origin/main^{commit}')
    $currentRevision = Get-GitRevision -ArgumentList @('rev-parse', '--verify', 'HEAD^{commit}')

    if (($localMainRevision -and $localMainRevision -eq $remoteRevision) -or
        ($currentRevision -and $currentRevision -eq $remoteRevision)) {
        Write-Ok "No updates available; origin/main is at $($remoteRevision.Substring(0, 8))."
    }
    else {
        Write-Info "An update is available on origin/main ($($remoteRevision.Substring(0, 8)))."
        Write-Info 'Check complete; no files were downloaded or changed.'
    }
}

function Update-Application {
    $currentBranch = Get-GitRevision -ArgumentList @('branch', '--show-current')
    if (-not $currentBranch) {
        $currentBranch = 'detached HEAD'
    }
    Write-Step "Pulling origin/main into the current checkout ($currentBranch)"
    Invoke-Checked -FilePath 'git.exe' -WorkingDirectory $RepoRoot -ArgumentList @('pull', 'origin', 'main')
    Write-Ok 'Application update from origin/main completed'
}

# ============================================================
# Dependency, database, and test maintenance
# ============================================================
function Install-OrUpdateApplication {
    $selectedInstallationType = $InstallationType
    $portableRuntimesReady = $false
    if (-not $selectedInstallationType) {
        Initialize-PortableRuntimes
        $portableRuntimesReady = $true
        $selectedInstallationType = Read-InstallationType
    }
    Install-ApplicationDependencies `
        -BuildFrontend $true `
        -InstallationType $selectedInstallationType `
        -PortableRuntimesReady:$portableRuntimesReady
    Initialize-Database
    if (Test-Path -LiteralPath $UvCacheDir) {
        Write-Step 'Pruning uv cache'
        $skipped = Remove-CacheContents -RootPath $UvCacheDir
        if ($skipped -gt 0) {
            Write-Info "$skipped uv cache entr$(if ($skipped -eq 1) { 'y' } else { 'ies' }) could not be removed; continuing"
        }
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
    Write-Step 'Synchronizing database schema'
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

# ============================================================
# User data and cleanup maintenance
# ============================================================
function Remove-ApplicationLogs {
    $logDir = Join-Path $RepoRoot 'app/resources/logs'
    if (-not (Test-Path -LiteralPath $logDir)) {
        Write-Info "Log directory does not exist: $logDir"
        return
    }
    Get-ChildItem -LiteralPath $logDir -Filter '*.log' -File | Remove-Item -Force
    Write-Ok 'Application logs removed'
}

function Remove-PathSafely {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Recurse
    )

    if (-not (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)) {
        return $true
    }
    try {
        Remove-Item -LiteralPath $Path -Force -Recurse:$Recurse -ErrorAction Stop
        return $true
    }
    catch {
        Write-Warning "Skipped locked or inaccessible cache item: $Path ($($_.Exception.Message))"
        return $false
    }
}

function Remove-CacheContents {
    param([Parameter(Mandatory = $true)][string]$RootPath)

    if (-not (Test-Path -LiteralPath $RootPath -ErrorAction SilentlyContinue)) {
        return 0
    }

    $skipped = 0
    $items = @(Get-ChildItem -LiteralPath $RootPath -Force -Recurse -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true })
    foreach ($item in $items) {
        if ($item.Name -eq '.gitkeep') {
            continue
        }
        if (-not (Remove-PathSafely -Path $item.FullName -Recurse:$item.PSIsContainer)) {
            $skipped++
        }
    }
    return $skipped
}

function Remove-PythonCaches {
    $skipped = 0
    $cacheDirectories = @(Get-ChildItem -LiteralPath $RepoRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object Name -eq '__pycache__' |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true })
    foreach ($directory in $cacheDirectories) {
        $skipped += Remove-CacheContents -RootPath $directory.FullName
        if (-not (Remove-PathSafely -Path $directory.FullName -Recurse)) {
            $skipped++
        }
    }
    return $skipped
}

function Remove-ToolCacheDirectories {
    $skipped = 0
    $cacheDirectories = @(Get-ChildItem -LiteralPath $RepoRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('.mypy_cache', '.ruff_cache') -or $_.Name -like '.pytest_cache*' } |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true })
    foreach ($directory in $cacheDirectories) {
        $skipped += Remove-CacheContents -RootPath $directory.FullName
        if (-not (Remove-PathSafely -Path $directory.FullName -Recurse)) {
            $skipped++
        }
    }
    return $skipped
}

function Clear-ApplicationCache {
    $skipped = 0
    foreach ($cacheRoot in @($RuntimeCacheDir, $TestCacheDir, $LegacyCacheDir)) {
        $skipped += Remove-CacheContents -RootPath $cacheRoot
    }
    $skipped += Remove-PythonCaches
    $skipped += Remove-ToolCacheDirectories
    if ($skipped -gt 0) {
        Write-Info "$skipped cache entr$(if ($skipped -eq 1) { 'y' } else { 'ies' }) could not be removed; rerun as administrator to remove locked entries"
    }
    Write-Ok 'Development caches and test artifacts cleared from runtimes/cache and app/tests/cache'
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

function Resolve-LauncherPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $expandedPath = [Environment]::ExpandEnvironmentVariables($Path.Trim())
    if ([IO.Path]::IsPathRooted($expandedPath)) {
        return [IO.Path]::GetFullPath($expandedPath)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $expandedPath))
}

function Test-TrackedApplicationFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf -ErrorAction SilentlyContinue)) {
        return $false
    }

    $repositoryRoot = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
    $candidate = [IO.Path]::GetFullPath((Get-Item -LiteralPath $Path).FullName)
    $repositoryPrefix = "$repositoryRoot\"
    if (-not $candidate.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }

    $gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -eq $gitCommand) {
        return $true
    }
    $relativePath = $candidate.Substring($repositoryPrefix.Length).Replace('\', '/')
    $null = & $gitCommand.Source -C $RepoRoot ls-files --error-unmatch -- $relativePath 2>$null
    return $LASTEXITCODE -eq 0
}

function Remove-UserDataPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{ Removed = 0; Skipped = 0 }
    }

    $removed = 0
    $skipped = 0
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) {
        if (Test-TrackedApplicationFile -Path $item.FullName) {
            Write-Info "Preserved tracked application file: $($item.FullName)"
            return [pscustomobject]@{ Removed = 0; Skipped = 1 }
        }
        if (Remove-PathSafely -Path $item.FullName) {
            return [pscustomobject]@{ Removed = 1; Skipped = 0 }
        }
        return [pscustomobject]@{ Removed = 0; Skipped = 1 }
    }

    $children = @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true })
    foreach ($child in $children) {
        if ($child.PSIsContainer) {
            $remaining = @(Get-ChildItem -LiteralPath $child.FullName -Force -ErrorAction SilentlyContinue)
            if ($remaining.Count -eq 0 -and (Remove-PathSafely -Path $child.FullName -Recurse)) {
                $removed++
            }
            continue
        }

        if (Test-TrackedApplicationFile -Path $child.FullName) {
            Write-Info "Preserved tracked application file: $($child.FullName)"
            $skipped++
            continue
        }
        if (Remove-PathSafely -Path $child.FullName) {
            $removed++
        }
        else {
            $skipped++
        }
    }

    if ($removed -gt 0 -or $skipped -gt 0) {
        Write-Info "${Label}: removed $removed item(s); preserved or skipped $skipped item(s)"
    }
    return [pscustomobject]@{ Removed = $removed; Skipped = $skipped }
}

function Confirm-RemoveAllData {
    param([switch]$Force)

    if ($Force) {
        return
    }
    if (-not [Environment]::UserInteractive -or [Console]::IsInputRedirected) {
        throw 'Remove All Data requires -Force when the console is not interactive.'
    }

    Write-Host 'This permanently deletes local user data, including the database, settings, logs, RAG data, exports, and state.' -ForegroundColor Yellow
    Write-Host 'Tracked application files are preserved.' -ForegroundColor Yellow
    $confirmation = Read-Host 'Type REMOVE ALL DATA to continue'
    if ($confirmation -cne 'REMOVE ALL DATA') {
        throw 'Remove All Data cancelled.'
    }
}

function Remove-AllData {
    Import-DotEnv
    Confirm-RemoveAllData -Force:$Force

    $resourceRootValue = if ($env:DILIGENT_RESOURCES_PATH) { $env:DILIGENT_RESOURCES_PATH } else { 'app/resources' }
    $resourceRoot = Resolve-LauncherPath -Path $resourceRootValue
    $databasePath = if ($env:DILIGENT_SQLITE_PATH) {
        Resolve-LauncherPath -Path $env:DILIGENT_SQLITE_PATH
    }
    else {
        Join-Path $resourceRoot 'database.db'
    }
    $keyMaterialPath = if ($env:DILIGENT_ACCESS_KEY_MATERIAL_FILE) {
        Resolve-LauncherPath -Path $env:DILIGENT_ACCESS_KEY_MATERIAL_FILE
    }
    else {
        Join-Path $resourceRoot 'access-key-material.json'
    }

    $targets = @(
        [pscustomobject]@{ Path = $EnvFile; Label = 'local settings' },
        [pscustomobject]@{ Path = $databasePath; Label = 'SQLite database' },
        [pscustomobject]@{ Path = ('{0}-wal' -f $databasePath); Label = 'SQLite write-ahead log' },
        [pscustomobject]@{ Path = ('{0}-shm' -f $databasePath); Label = 'SQLite shared-memory file' },
        [pscustomobject]@{ Path = $keyMaterialPath; Label = 'access-key material' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'logs'); Label = 'application logs' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'models/embeddings'); Label = 'generated embedding models' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'sources/archives'); Label = 'downloaded source archives' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'sources/documents'); Label = 'user source documents' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'sources/vectors'); Label = 'generated vector index' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'exports'); Label = 'user exports' },
        [pscustomobject]@{ Path = (Join-Path $resourceRoot 'state'); Label = 'runtime state' }
    )

    $removed = 0
    $skipped = 0
    foreach ($target in $targets) {
        $result = Remove-UserDataPath -Path $target.Path -Label $target.Label
        $removed += $result.Removed
        $skipped += $result.Skipped
    }

    if ($env:EMBEDDED_DATABASE -and $env:EMBEDDED_DATABASE -notmatch '^(?i:true)$') {
        Write-Info 'External database mode is configured; no external database was modified.'
    }
    if ($skipped -gt 0) {
        Write-Info "$skipped item(s) were preserved or could not be removed. Review the messages above before restarting."
    }
    Write-Ok "All removable local user data was cleared ($removed item(s) removed); application files were preserved"
}

# ============================================================
# Desktop release management
# ============================================================
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
    param([switch]$Interactive)

    if ($OfflineWebView2 -and $Action -ne 'BuildDesktopRelease') {
        throw '-OfflineWebView2 is valid only with BuildDesktopRelease'
    }
    if ($OfflineWebView2 -and $DesktopTarget -eq 'Portable') {
        throw '-OfflineWebView2 requires an MSI target'
    }
    if ($AllDesktopReleases -and $Action -ne 'RemoveDesktopRelease' -and -not $Interactive) {
        throw '-AllDesktopReleases is valid only with RemoveDesktopRelease'
    }
    if ($Action -eq 'RemoveDesktopRelease' -and -not $AllDesktopReleases -and -not $Version -and -not $Interactive) {
        throw 'RemoveDesktopRelease requires -Version or -AllDesktopReleases'
    }
}

function Get-DesktopReleaseArtifactPaths {
    param([Parameter(Mandatory = $true)][string]$Version)
    $prefix = "DILIGENT-v$Version-windows-x64"
    return [pscustomobject]@{
        Portable = Join-Path $DesktopArtifactsDir "$prefix-portable.exe"
        Msi = Join-Path $DesktopArtifactsDir "$prefix.msi"
        Checksum = Join-Path $DesktopArtifactsDir "$prefix.sha256"
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
        $migrationDatabase = Join-Path $dataRoot 'resources/database.db'
        Invoke-Checked -FilePath $VenvPython -WorkingDirectory $ServerDir -ArgumentList @(
            '-c',
            'import sqlite3,sys; from repositories.database.migrations import HEAD_REVISION; connection=sqlite3.connect(sys.argv[1]); heads={row[0] for row in connection.execute("select version_num from alembic_version")}; connection.close(); assert heads == {HEAD_REVISION}, f"database heads {heads!r} != {HEAD_REVISION!r}"',
            $migrationDatabase
        )
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
    $paths = Get-DesktopReleaseArtifactPaths -Version $Version
    $portable = $paths.Portable
    $msi = $paths.Msi
    $rawExe = Get-ChildItem -LiteralPath (Join-Path $CargoTargetDir 'release') -Filter '*.exe' -File -ErrorAction SilentlyContinue | Where-Object Name -notmatch 'uninstall' | Select-Object -First 1
    if ($DesktopTarget -in @('Portable', 'All')) {
        if ($null -eq $rawExe) { throw 'Tauri raw executable was not found' }
        if ((Test-Path -LiteralPath $portable) -and -not $Force) { throw "Release already exists: $portable (use -Force to replace it)" }
        Copy-Item -LiteralPath $rawExe.FullName -Destination $portable -Force
        Test-PortableDesktopArtifact -Path $portable
    }
    if ($DesktopTarget -in @('Msi', 'All')) {
        $builtMsi = Get-ChildItem -LiteralPath (Join-Path $CargoTargetDir 'release/bundle/msi') -Filter '*.msi' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like ("*_{0}_*.msi" -f $Version) } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -eq $builtMsi) { throw 'Tauri MSI artifact was not found' }
        if ((Test-Path -LiteralPath $msi) -and -not $Force) { throw "Release already exists: $msi (use -Force to replace it)" }
        Copy-Item -LiteralPath $builtMsi.FullName -Destination $msi -Force
        Test-MsiMetadata -Path $msi
    }
    Write-DesktopChecksums -Version $Version
}

function Write-DesktopChecksums {
    param([Parameter(Mandatory = $true)][string]$Version)
    $paths = Get-DesktopReleaseArtifactPaths -Version $Version
    $entries = @()
    foreach ($artifact in @(
        [pscustomobject]@{ Name = (Split-Path -Leaf $paths.Portable); Path = $paths.Portable }
        [pscustomobject]@{ Name = (Split-Path -Leaf $paths.Msi); Path = $paths.Msi }
    )) {
        if (Test-Path -LiteralPath $artifact.Path -PathType Leaf) {
            $entries += [pscustomobject]@{
                Name = $artifact.Name
                Hash = (Get-FileHash -LiteralPath $artifact.Path -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }
    if ($entries.Count -eq 0) {
        throw "No desktop release artifacts exist for version $Version"
    }
    $lines = foreach ($entry in $entries) {
        "SHA256  $($entry.Name)"
        "$($entry.Hash)  $($entry.Name)"
    }
    Set-Content -LiteralPath $paths.Checksum -Value $lines -Encoding ascii
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

function Remove-DesktopGeneratedState {
    foreach ($target in @($DesktopBuildDir, $CargoTargetDir, (Join-Path $DesktopDir 'node_modules'))) {
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    if (Test-Path -LiteralPath $DesktopGeneratedDir) {
        Get-ChildItem -LiteralPath $DesktopGeneratedDir -Force |
            Where-Object { $_.Name -ne '.gitkeep' } |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    }
}

function Remove-DesktopRelease {
    param(
        [ValidateSet('Portable', 'Msi', 'Checksum', 'All')]
        [string]$ArtifactTarget,
        [switch]$Interactive
    )

    Assert-DesktopParameterContract -Interactive:$Interactive
    $selectedTarget = if ($ArtifactTarget) { $ArtifactTarget } else { $DesktopTarget }
    $removeBuildState = $AllDesktopReleases -or $selectedTarget -eq 'All'
    if ($AllDesktopReleases) {
        if (Test-Path -LiteralPath $DesktopArtifactsDir) { Get-ChildItem -LiteralPath $DesktopArtifactsDir -File | Where-Object Name -match '^DILIGENT-v\d+\.\d+\.\d+-windows-x64' | Remove-Item -Force }
    }
    else {
        $resolvedVersion = if ($Version) { $Version } else { Resolve-DesktopReleaseVersion }
        $paths = Get-DesktopReleaseArtifactPaths -Version $resolvedVersion
        $targets = switch ($selectedTarget) {
            'Portable' { @($paths.Portable) }
            'Msi' { @($paths.Msi) }
            'Checksum' { @($paths.Checksum) }
            'All' { @($paths.Portable, $paths.Msi, $paths.Checksum) }
        }
        foreach ($target in $targets) {
            if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force }
        }
        if ($selectedTarget -in @('Portable', 'Msi')) {
            if ((Test-Path -LiteralPath $paths.Portable -PathType Leaf) -or (Test-Path -LiteralPath $paths.Msi -PathType Leaf)) {
                Write-DesktopChecksums -Version $resolvedVersion
            }
            elseif (Test-Path -LiteralPath $paths.Checksum -PathType Leaf) {
                Remove-Item -LiteralPath $paths.Checksum -Force
            }
        }
    }
    if ($removeBuildState) {
        Remove-DesktopGeneratedState
        Write-Ok 'Selected desktop release artifacts and generated desktop build state removed; installed applications and user data were preserved'
    }
    else {
        Write-Ok 'Selected desktop release artifact(s) removed; the checksum manifest was synchronized when applicable and other generated state was preserved'
    }
}

# ============================================================
# Interactive menu and action dispatch
# ============================================================
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

function Write-MenuSection([string]$Title) {
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text $Title -Color Yellow
}

function Show-MainMenu {
    Write-Host ''
    Write-MenuRule
    Write-MenuLine -Text 'DILIGENT  /  CLINICAL COPILOT' -Color Cyan
    Write-MenuLine -Text 'Local development and maintenance console' -Color DarkGray
    Write-MenuRule
    Write-MenuSection -Title 'APPLICATION'
    Write-MenuOption -Number '1.' -Label 'Launch application' -Description 'Start local services'
    Write-MenuSection -Title 'SOURCE CONTROL'
    Write-MenuOption -Number '2.' -Label 'Check for updates' -Description 'Report origin/main status only'
    Write-MenuOption -Number '3.' -Label 'Update application' -Description 'Pull latest changes from origin/main'
    Write-MenuSection -Title 'SETUP & MAINTENANCE'
    Write-MenuOption -Number '4.' -Label 'Install dependencies' -Description 'Sync runtimes + packages'
    Write-MenuOption -Number '5.' -Label 'Rebuild frontend' -Description 'Recreate Angular production output'
    Write-MenuOption -Number '6.' -Label 'Initialize database' -Description 'Prepare local data store'
    Write-MenuOption -Number '7.' -Label 'Run test suite' -Description 'Execute project checks'
    Write-MenuSection -Title 'DATA & CLEANUP'
    Write-MenuOption -Number '8.' -Label 'Remove logs' -Description 'Delete application logs'
    Write-MenuOption -Number '9.' -Label 'Clear cache' -Description 'Remove temporary caches'
    Write-MenuOption -Number '10.' -Label 'Remove all data' -Description 'Delete local user data only'
    Write-MenuOption -Number '11.' -Label 'Uninstall application' -Description 'Remove generated dependencies'
    Write-MenuSection -Title 'DESKTOP RELEASE'
    Write-MenuOption -Number '12.' -Label 'Create release artifacts' -Description 'Choose package, manifest, or all'
    Write-MenuOption -Number '13.' -Label 'Remove release artifacts' -Description 'Choose artifact(s) or all versions'
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text '14. Exit'
    Write-MenuRule
}

function Wait-ForMenu {
    Write-Host ''
    Write-Host 'Press any key to return to menu...'
    [Console]::ReadKey($true) | Out-Null
}

function Read-DesktopArtifactSelection {
    param([ValidateSet('Create', 'Remove')][string]$Operation)

    Write-Host ''
    if ($Operation -eq 'Create') {
        Write-Host 'CREATE DESKTOP RELEASE ARTIFACTS' -ForegroundColor Yellow
        Write-Host '  [1] Portable executable'
        Write-Host '  [2] MSI installer'
        Write-Host '  [3] SHA-256 manifest (from existing artifacts)'
        Write-Host '  [4] All distribution artifacts'
        Write-Host '  [5] Back'
        $selection = (Read-Host '  Select an artifact to create [1-5]').Trim()
        switch ($selection) {
            '1' { return [pscustomobject]@{ Target = 'Portable'; AllVersions = $false } }
            '2' { return [pscustomobject]@{ Target = 'Msi'; AllVersions = $false } }
            '3' { return [pscustomobject]@{ Target = 'Checksum'; AllVersions = $false } }
            '4' { return [pscustomobject]@{ Target = 'All'; AllVersions = $false } }
            '5' { return $null }
            default { throw 'Invalid selection. Enter a number from 1 through 5.' }
        }
    }

    Write-Host 'REMOVE DESKTOP RELEASE ARTIFACTS' -ForegroundColor Yellow
    Write-Host '  [1] Portable executable (selected version)'
    Write-Host '  [2] MSI installer (selected version)'
    Write-Host '  [3] SHA-256 manifest (selected version)'
    Write-Host '  [4] All artifacts for one version'
    Write-Host '  [5] All versions and artifacts'
    Write-Host '  [6] Back'
    $selection = (Read-Host '  Select artifacts to remove [1-6]').Trim()
    switch ($selection) {
        '1' { return [pscustomobject]@{ Target = 'Portable'; AllVersions = $false } }
        '2' { return [pscustomobject]@{ Target = 'Msi'; AllVersions = $false } }
        '3' { return [pscustomobject]@{ Target = 'Checksum'; AllVersions = $false } }
        '4' { return [pscustomobject]@{ Target = 'All'; AllVersions = $false } }
        '5' { return [pscustomobject]@{ Target = 'All'; AllVersions = $true } }
        '6' { return $null }
        default { throw 'Invalid selection. Enter a number from 1 through 6.' }
    }
}

function Invoke-CreateDesktopReleaseMenu {
    $selection = Read-DesktopArtifactSelection -Operation 'Create'
    if ($null -eq $selection) { return }

    if ($selection.Target -eq 'Checksum') {
        $resolvedVersion = Resolve-DesktopReleaseVersion
        $paths = Get-DesktopReleaseArtifactPaths -Version $resolvedVersion
        if (-not (Test-Path -LiteralPath $paths.Portable -PathType Leaf) -and -not (Test-Path -LiteralPath $paths.Msi -PathType Leaf)) {
            throw "Cannot create a checksum manifest because no release artifact exists for version $resolvedVersion"
        }
        Write-DesktopChecksums -Version $resolvedVersion
        Write-Ok "SHA-256 manifest created for desktop release $resolvedVersion"
        return
    }

    $script:DesktopTarget = $selection.Target
    Build-DesktopRelease
}

function Invoke-RemoveDesktopReleaseMenu {
    $selection = Read-DesktopArtifactSelection -Operation 'Remove'
    if ($null -eq $selection) { return }

    $script:DesktopTarget = $selection.Target
    $script:AllDesktopReleases = [bool]$selection.AllVersions
    Remove-DesktopRelease -Interactive
}

if ($Action) {
    switch ($Action) {
        'Launch' { Start-Application }
        'Install' { Install-OrUpdateApplication }
        'RebuildFrontend' { Rebuild-Frontend }
        'InitializeDatabase' { Initialize-Database }
        'Test' { Invoke-TestSuite }
        'Uninstall' { Uninstall-Application }
        'Update' { Update-Application }
        'CheckForUpdates' { Check-ForUpdates }
        'RemoveAllData' { Remove-AllData }
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
    $rawSelection = Read-Host 'Select an option (1-14)'
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
            '^2$' { Check-ForUpdates }
            '^3$' { Update-Application }
            '^4$' { Install-OrUpdateApplication }
            '^5$' { Rebuild-Frontend }
            '^6$' { Initialize-Database }
            '^7$' { Invoke-TestSuite }
            '^8$' { Remove-ApplicationLogs }
            '^9$' { Clear-ApplicationCache }
            '^10$' { Remove-AllData }
            '^11$' { Uninstall-Application }
            '^12$' { Invoke-CreateDesktopReleaseMenu }
            '^13$' { Invoke-RemoveDesktopReleaseMenu }
            '^14$' { exit 0 }
            default {
                Write-Host '[ERROR] Select a number from 1 through 14.' -ForegroundColor Red
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

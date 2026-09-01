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
$script:CargoTargetDir = Join-Path $RepoRoot 'assets/QA/desktop-cargo-target'
$script:MypyCacheDir = Join-Path $TestCacheDir 'mypy'
$script:NpmCacheDir = Join-Path $RuntimeCacheDir 'npm'
$script:PipCacheDir = Join-Path $RuntimeCacheDir 'pip'
$script:PlaywrightCacheDir = Join-Path $RuntimeCacheDir 'playwright'
$script:PytestCacheDir = Join-Path $TestCacheDir 'pytest'
$script:PythonBytecodeCacheDir = Join-Path $RuntimeCacheDir 'python'
$script:RuffCacheDir = Join-Path $TestCacheDir 'ruff'
$script:UvCacheDir = Join-Path $RepoRoot 'assets/QA/desktop-release-uv-cache'
$script:EnvFile = Join-Path $RepoRoot 'settings/.env'
$script:EnvExample = Join-Path $RepoRoot 'settings/.env.example'
$script:PythonVersion = '3.14.2'
$script:NodeVersion = '22.13.0'
$script:UvVersion = '0.11.30'
$script:RustVersion = '1.95.0'
$script:PythonArchiveSha256 = 'f05e28d161c6b15af64a7cb7f08b4a22b3a6b03eee71baee24ea557b3bdd5798'
$script:NodeArchiveSha256 = 'b0feb09ebf41328628e7383f7a092fb7342ce1e05c867a90cf8f1379205a8429'
$script:UvArchiveSha256 = 'be8d78c992312212e5cc05e9f9de3fa996db73b7c86a186dfb9231eb9f91d33e'
$script:DesktopDir = Join-Path $RepoRoot 'app/desktop'
$script:DesktopTauriDir = Join-Path $DesktopDir 'src-tauri'
$script:DesktopBuildDir = Join-Path $RepoRoot 'assets/QA/desktop-release-staging'
$script:DesktopGeneratedDir = Join-Path $DesktopTauriDir 'generated'
$script:DesktopArtifactsDir = Join-Path $RepoRoot 'release'
$script:DesktopStageRoot = Join-Path $DesktopBuildDir 'staging'
$script:DesktopNpmCmd = Join-Path $NodeDir 'npm.cmd'
$script:DesktopTauriCli = Join-Path $DesktopDir 'node_modules/.bin/tauri.cmd'
$script:DesktopFrontendOutputDir = $null
$script:PyInstallerVersion = '6.21.0'
$script:DesktopTargetTriple = 'x86_64-pc-windows-msvc'
$script:NextProgressId = 1
$script:ActiveProgressActivities = [Collections.Generic.Dictionary[int, string]]::new()
$script:LauncherInteractive = -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected

# ============================================================
# Shared output and process helpers
# ============================================================
function Write-Step([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Info([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[INFO] $Message" -ForegroundColor DarkCyan
}

function Write-Fatal([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[FATAL] $Message" -ForegroundColor Red
}

function Start-LauncherProgress {
    param([Parameter(Mandatory = $true)][string]$Activity, [Parameter(Mandatory = $true)][string]$Status)
    $id = $script:NextProgressId++
    $script:ActiveProgressActivities[$id] = $Activity
    if ($script:LauncherInteractive) { Write-Progress -Id $id -Activity $Activity -Status $Status }
    return $id
}

function Update-LauncherProgress {
    param(
        [Parameter(Mandatory = $true)][int]$Id,
        [Parameter(Mandatory = $true)][string]$Activity,
        [Parameter(Mandatory = $true)][string]$Status,
        [Nullable[int]]$PercentComplete
    )
    if (-not $script:ActiveProgressActivities.ContainsKey($Id)) { return }
    $activity = $script:ActiveProgressActivities[$Id]
    $progress = @{ Id = $Id; Activity = $activity; Status = $Status }
    if ($null -ne $PercentComplete) { $progress.PercentComplete = $PercentComplete }
    if ($script:LauncherInteractive) { Write-Progress @progress }
}

function Complete-LauncherProgress([int]$Id) {
    if ($script:ActiveProgressActivities.ContainsKey($Id)) {
        $activity = $script:ActiveProgressActivities[$Id]
        try {
            if ($script:LauncherInteractive) {
                try { Write-Progress -Id $Id -Activity $activity -Completed } catch { }
            }
        }
        finally {
            [void]$script:ActiveProgressActivities.Remove($Id)
        }
    }
}

function Clear-LauncherProgress {
    foreach ($id in @($script:ActiveProgressActivities.Keys)) {
        Complete-LauncherProgress -Id $id
    }
}

function Invoke-TrackedLauncherAction {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Operation
    )
    Write-Step "Starting $Name"
    try {
        & $Operation
        Write-Ok "$Name completed"
    }
    catch {
        Write-Fatal "$Name failed: $($_.Exception.Message)"
        throw
    }
    finally {
        Clear-LauncherProgress
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $RepoRoot
    )

    $display = "$FilePath " + ($ArgumentList -join ' ')
    Write-Step "Running $display"
    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
    }
    finally {
        Pop-Location
    }
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode`: $display"
    }
    Write-Ok "Completed $display"
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [string]$ExpectedSha256
    )
    $prevProgress = $ProgressPreference
    $activity = "DILIGENT: download and extract $([IO.Path]::GetFileName($ArchivePath))"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Downloading $Uri"
    try {
        $ProgressPreference = 'SilentlyContinue'
        New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivePath) -Force | Out-Null
        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
        Invoke-WebRequest -Uri $Uri -OutFile $ArchivePath
        $ProgressPreference = $prevProgress
        if ($ExpectedSha256) {
            Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Verifying archive checksum'
            $actualSha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
                throw "Downloaded archive checksum mismatch for $Uri"
            }
        }
        Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Extracting archive'
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
        Remove-Item -LiteralPath $ArchivePath -Force
    }
    finally {
        $ProgressPreference = $prevProgress
        Complete-LauncherProgress $progressId
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
    $activity = "DILIGENT: wait for health $Url"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Waiting up to $Attempts attempts"
    try {
        for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
            Update-LauncherProgress -Id $progressId -Activity $activity -Status "Attempt $attempt of $Attempts"
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
        Complete-LauncherProgress $progressId
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
                -DestinationPath $nodeStageDir `
                -ExpectedSha256 $NodeArchiveSha256

            $expandedNodeDir = Join-Path $nodeStageDir "node-v$NodeVersion-win-x64"
            if (-not (Test-Path -LiteralPath (Join-Path $expandedNodeDir 'node.exe'))) {
                throw "Downloaded Node.js archive did not contain the expected runtime"
            }

            [void](Remove-LauncherPath -Path $NodeDir -Activity 'DILIGENT: replace portable Node.js runtime' -Strict)
            Move-Item -LiteralPath $expandedNodeDir -Destination $NodeDir
        }
        finally {
            if (Test-Path -LiteralPath $nodeStageDir) {
                [void](Remove-LauncherPath -Path $nodeStageDir -Activity 'DILIGENT: remove Node.js staging directory')
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
            -DestinationPath $PythonDir `
            -ExpectedSha256 $PythonArchiveSha256
    }

    Patch-PythonPth -Path $PythonPth
    $pythonVersionFound = Invoke-PythonVersionCheck -PythonExecutable $PythonExe
    Write-Ok "Python ready: $pythonVersionFound"

    $uvNeedsInstall = -not (Test-Path -LiteralPath $UvExe)
    if (-not $uvNeedsInstall) {
        $installedUv = (& $UvExe --version).Trim()
        $uvNeedsInstall = $installedUv -notmatch "^uv $([regex]::Escape($UvVersion))\b"
        if ($uvNeedsInstall) {
            Write-Info "Replacing unsupported portable uv $installedUv with $UvVersion"
        }
    }
    if ($uvNeedsInstall) {
        $uvTarget = 'uv-x86_64-pc-windows-msvc.zip'
        $uvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$uvTarget"
        Write-Info "Downloading $uvUrl"
        Invoke-DownloadAndExtract `
            -Uri $uvUrl `
            -ArchivePath (Join-Path $UvDir 'uv.zip') `
            -DestinationPath $UvDir `
            -ExpectedSha256 $UvArchiveSha256
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
        [switch]$PortableRuntimesReady,
        [switch]$DesktopBuild
    )

    if (-not $PortableRuntimesReady) {
        Initialize-PortableRuntimes
    }
    Import-DotEnv
    Set-LauncherEnvironment

    Write-Step 'Installing Python dependencies'
    $syncArguments = @('sync', '--locked', '--python', $PythonExe)
    if ($DesktopBuild) {
        $syncArguments += @('--group', 'desktop-build')
    }
    if ($InstallationType -eq 'Development') {
        $syncArguments += '--all-extras'
    }
    try {
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArguments -WorkingDirectory $ServerDir
    }
    catch {
        Write-Info 'Recreating a virtual environment that may reference an older repository location'
        if (Test-Path -LiteralPath $VenvDir) {
            [void](Remove-LauncherPath -Path $VenvDir -Activity 'DILIGENT: recreate Python environment' -Strict)
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
    $buildArguments = @('run', 'build')
    if ($script:DesktopFrontendOutputDir) {
        $buildArguments += @('--', '--output-path', $script:DesktopFrontendOutputDir)
    }
    Invoke-Checked -FilePath $NpmCmd -ArgumentList $buildArguments -WorkingDirectory $ClientDir
    $outputRoot = if ($script:DesktopFrontendOutputDir) { $script:DesktopFrontendOutputDir } else { Join-Path $ClientDir 'dist' }
    if (-not (Test-Path -LiteralPath (Join-Path $outputRoot 'browser/index.html') -PathType Leaf)) {
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
    if (-not $currentBranch) { throw 'Update requires a non-detached Git checkout.' }
    if ($currentBranch -ne 'main') {
        throw "Update requires the main branch to be checked out; current branch is '$currentBranch'. No files were changed."
    }
    $statusOutput = @(& git.exe -C $RepoRoot status --porcelain 2>$null)
    $statusExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    if ($statusExitCode -ne 0) { throw 'Unable to inspect the Git working tree before updating.' }
    $changes = @($statusOutput | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($changes.Count -gt 0) {
        throw 'Update requires a clean Git working tree. Commit or safely preserve local changes before retrying.'
    }
    Write-Step 'Pulling origin/main (fast-forward only)'
    Invoke-Checked -FilePath 'git.exe' -WorkingDirectory $RepoRoot -ArgumentList @('pull', '--ff-only', 'origin', 'main')
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
    $logs = @(Get-ChildItem -LiteralPath $logDir -Filter '*.log' -File -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    if ($logs.Count -eq 0) {
        Write-Info 'No application log files found'
        return
    }
    $skipped = 0
    $progressId = Start-LauncherProgress -Activity 'DILIGENT: remove application logs' -Status "0 of $($logs.Count) files"
    try {
        for ($index = 0; $index -lt $logs.Count; $index++) {
            $log = $logs[$index]
            Update-LauncherProgress -Id $progressId -Activity 'DILIGENT: remove application logs' -Status "$($index + 1) of $($logs.Count): $($log.Name)" -PercentComplete ([int](($index + 1) * 100 / $logs.Count))
            if (-not (Remove-PathSafely -Path $log.FullName)) {
                $skipped++
            }
        }
    }
    finally {
        Complete-LauncherProgress $progressId
    }
    Write-Ok "Removed $($logs.Count - $skipped) application log file(s); skipped $skipped locked or inaccessible file(s)"
}

function Remove-LauncherPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$KeepRoot,
        [string[]]$PreserveNames = @('.gitkeep'),
        [string[]]$PreservePaths = @(),
        [switch]$Strict,
        [switch]$WhatIf,
        [string]$Activity = 'DILIGENT: remove files'
    )

    $fullPath = [IO.Path]::GetFullPath($Path)
    $removed = [Collections.Generic.List[string]]::new()
    $skipped = [Collections.Generic.List[string]]::new()
    $preserved = [Collections.Generic.List[string]]::new()
    $enumerationErrors = [Collections.Generic.List[string]]::new()
    $result = [ordered]@{
        Target = $fullPath
        Path = $fullPath
        Planned = 0
        PlannedCount = 0
        Removed = 0
        RemovedCount = 0
        RemovedPaths = $removed
        Preserved = 0
        PreservedEntries = $preserved
        Skipped = 0
        SkippedPaths = $skipped
        EnumerationErrors = $enumerationErrors
        WhatIf = [bool]$WhatIf
    }

    try {
        $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
            return [pscustomobject]$result
        }
        [void]$enumerationErrors.Add("$fullPath ($($_.Exception.Message))")
        Write-Info "Skipped inaccessible path: $fullPath ($($_.Exception.Message))"
        if ($Strict) { throw }
        return [pscustomobject]$result
    }

    $entries = if ($item.PSIsContainer) {
        $errors = @()
        $found = @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -ErrorAction SilentlyContinue -ErrorVariable errors)
        foreach ($errorRecord in $errors) {
            [void]$enumerationErrors.Add("$($errorRecord.Exception.Message)")
            Write-Info "Skipped inaccessible path below $fullPath ($($errorRecord.Exception.Message))"
        }
        if (-not $KeepRoot) { $found += $item }
        $found
    }
    else { @($item) }

    $protectedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $preservedPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($preservePath in $PreservePaths) {
        if (-not [string]::IsNullOrWhiteSpace($preservePath)) {
            [void]$preservedPaths.Add([IO.Path]::GetFullPath($preservePath))
        }
    }
    foreach ($entry in @($entries)) {
        if ($entry.Name -in $PreserveNames -or $preservedPaths.Contains($entry.FullName)) {
            [void]$preservedPaths.Add($entry.FullName)
            [void]$preserved.Add($entry.FullName)
            $ancestor = [IO.Path]::GetDirectoryName($entry.FullName)
            while ($ancestor -and $ancestor.StartsWith($item.FullName.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                [void]$protectedDirectories.Add($ancestor)
                $ancestor = [IO.Path]::GetDirectoryName($ancestor)
            }
        }
    }

    $candidates = @($entries |
        Where-Object { -not $preservedPaths.Contains($_.FullName) -and -not $protectedDirectories.Contains($_.FullName) } |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    $result.Planned = $candidates.Count
    $result.PlannedCount = $candidates.Count
    $result.Preserved = $preserved.Count
    $progressId = $null
    try {
        if ($candidates.Count -gt 0) {
            $progressId = Start-LauncherProgress -Activity $Activity -Status "0 of $($candidates.Count) items"
        }
        for ($index = 0; $index -lt $candidates.Count; $index++) {
            $entry = $candidates[$index]
            if ($null -ne $progressId) {
                Update-LauncherProgress -Id $progressId -Activity $Activity -Status "$($index + 1) of $($candidates.Count): $($entry.Name)" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $candidates.Count)))
            }
            if ($WhatIf) { continue }
            try {
                Remove-Item -LiteralPath $entry.FullName -Force -Confirm:$false -ErrorAction Stop
                [void]$removed.Add($entry.FullName)
            }
            catch {
                [void]$skipped.Add("$($entry.FullName) ($($_.Exception.Message))")
                Write-Info "Skipped locked or inaccessible path: $($entry.FullName) ($($_.Exception.Message))"
            }
        }
    }
    finally {
        if ($null -ne $progressId) { Complete-LauncherProgress -Id $progressId }
    }
    $result.Removed = $removed.Count
    $result.RemovedCount = $removed.Count
    $result.Skipped = $skipped.Count
    if ($Strict -and ($skipped.Count -gt 0 -or $enumerationErrors.Count -gt 0)) {
        throw "Removal of '$fullPath' was incomplete. Skipped $($skipped.Count) item(s) and encountered $($enumerationErrors.Count) enumeration error(s)."
    }
    return [pscustomobject]$result
}

function Remove-PathSafely {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Recurse
    )
    $result = Remove-LauncherPath -Path $Path -Activity "DILIGENT: remove $([IO.Path]::GetFileName($Path))"
    return $result.Skipped -eq 0 -and $result.EnumerationErrors.Count -eq 0
}

function Remove-CacheContents {
    param([Parameter(Mandatory = $true)][string]$RootPath)

    if (-not (Test-Path -LiteralPath $RootPath -ErrorAction SilentlyContinue)) {
        return 0
    }

    $skipped = 0
    $enumerationErrors = @()
    $items = @(Get-ChildItem -LiteralPath $RootPath -Force -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors |
        Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    $skipped += $enumerationErrors.Count
    foreach ($enumerationError in $enumerationErrors) {
        Write-Warning "Skipped inaccessible cache item below $RootPath ($($enumerationError.Exception.Message))"
    }
    for ($index = 0; $index -lt $items.Count; $index++) {
        $item = $items[$index]
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
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    foreach ($directory in $cacheDirectories) {
        if (-not (Remove-PathSafely -Path $directory.FullName -Recurse)) { $skipped++ }
    }
    return $skipped
}

function Remove-ToolCacheDirectories {
    $skipped = 0
    $cacheDirectories = @(Get-ChildItem -LiteralPath $RepoRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('.mypy_cache', '.ruff_cache') -or $_.Name -like '.pytest_cache*' } |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    foreach ($directory in $cacheDirectories) {
        $skipped += Remove-CacheContents -RootPath $directory.FullName
        if (Test-Path -LiteralPath $directory.FullName -PathType Container -ErrorAction SilentlyContinue) {
            $remaining = @(Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction SilentlyContinue)
            if ($remaining.Count -eq 0 -and -not (Remove-PathSafely -Path $directory.FullName -Recurse)) {
                $skipped++
            }
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
    $skipped = 0
    $progressId = Start-LauncherProgress -Activity 'DILIGENT: uninstall application' -Status "0 of $($targets.Count) paths"
    try {
        for ($index = 0; $index -lt $targets.Count; $index++) {
            $target = $targets[$index]
            if (Test-Path -LiteralPath $target) {
                Update-LauncherProgress -Id $progressId -Activity 'DILIGENT: uninstall application' -Status "$($index + 1) of $($targets.Count): $target" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $targets.Count)))
                if (-not (Remove-PathSafely -Path $target -Recurse)) {
                    $skipped++
                }
            }
        }
    }
    finally {
        Complete-LauncherProgress $progressId
    }

    Remove-PythonCaches
    if ($skipped -gt 0) {
        Write-Warning "$skipped uninstall target(s) could not be removed."
    }
    Write-Ok 'Application runtimes, dependencies, and build outputs removed where permitted. Dependency lockfiles and user data were preserved.'
}

function Resolve-LauncherPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $expandedPath = [Environment]::ExpandEnvironmentVariables($Path.Trim())
    if ([IO.Path]::IsPathRooted($expandedPath)) {
        return [IO.Path]::GetFullPath($expandedPath)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $expandedPath))
}

function Get-TrackedApplicationFilesUnderPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -eq $gitCommand) {
        return [pscustomobject]@{ Available = $false; Files = @() }
    }

    $repositoryRoot = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
    $candidate = [IO.Path]::GetFullPath($Path)
    $repositoryPrefix = "$repositoryRoot\"
    if (-not $candidate.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ Available = $true; Files = @() }
    }

    $relativePath = $candidate.Substring($repositoryPrefix.Length).Replace('\', '/')
    if ([string]::IsNullOrWhiteSpace($relativePath)) {
        return [pscustomobject]@{ Available = $false; Files = @() }
    }
    $tracked = @(& $gitCommand.Source -C $RepoRoot ls-files -- $relativePath "$relativePath/**" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return [pscustomobject]@{ Available = $false; Files = @() }
    }
    $trackedFiles = @($tracked | ForEach-Object {
        $relativeFile = ([string]$_).Trim()
        if (-not [string]::IsNullOrWhiteSpace($relativeFile)) {
            [IO.Path]::GetFullPath((Join-Path $RepoRoot ($relativeFile -replace '/', '\')))
        }
    })
    return [pscustomobject]@{ Available = $true; Files = $trackedFiles }
}

function Remove-UserDataPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    try {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    } catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
            return [pscustomobject]@{ Removed = 0; Skipped = 0 }
        }
        Write-Info "${Label}: path was inaccessible; nothing was removed ($($_.Exception.Message))"
        return [pscustomobject]@{ Removed = 0; Skipped = 1 }
    }

    $trackedFiles = @()
    $repositoryRoot = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
    if ($item.FullName.Equals($repositoryRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $item.FullName.StartsWith("$repositoryRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        $tracking = Get-TrackedApplicationFilesUnderPath -Path $item.FullName
        if (-not $tracking.Available) {
            Write-Info "${Label}: tracked-file verification was unavailable; nothing was removed."
            return [pscustomobject]@{ Removed = 0; Skipped = 1 }
        }
        $trackedFiles = @($tracking.Files)
    }

    $result = Remove-LauncherPath -Path $item.FullName -KeepRoot:$item.PSIsContainer `
        -PreserveNames @('.gitkeep') -PreservePaths $trackedFiles -Activity "DILIGENT: remove $Label"
    foreach ($preservedPath in @($result.PreservedEntries)) {
        Write-Info "Preserved application file: $preservedPath"
    }
    if ($result.Skipped -gt 0 -or $result.EnumerationErrors.Count -gt 0) {
        Write-Info "${Label}: removed $($result.Removed) item(s); skipped or preserved $($result.Skipped + $result.EnumerationErrors.Count) item(s)"
    }
    return [pscustomobject]@{
        Removed = $result.Removed
        Skipped = $result.Skipped + $result.EnumerationErrors.Count
    }
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
    $confirmation = ([string](Read-Host 'Continue removing all local user data? [y/N]')).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
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

function Assert-DesktopReleaseVersion {
    param([Parameter(Mandatory = $true)][string]$ExpectedVersion)

    $jsonManifests = @(
        (Join-Path $ClientDir 'package.json'),
        (Join-Path $DesktopDir 'package.json')
    )
    foreach ($path in $jsonManifests) {
        $actual = (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).version
        if ($actual -ne $ExpectedVersion) {
            throw "Version mismatch in ${path}: expected $ExpectedVersion, found $actual"
        }
    }

    $textManifests = @(
        [pscustomobject]@{ Path = (Join-Path $ServerDir 'pyproject.toml'); Pattern = '(?m)^version\s*=\s*"([^"]+)"' },
        [pscustomobject]@{ Path = (Join-Path $DesktopTauriDir 'Cargo.toml'); Pattern = '(?m)^version\s*=\s*"([^"]+)"' },
        [pscustomobject]@{ Path = (Join-Path $DesktopTauriDir 'tauri.conf.json'); Pattern = '"version"\s*:\s*"([^"]+)"' },
        [pscustomobject]@{ Path = (Join-Path $RepoRoot 'app/shared/openapi.json'); Pattern = '"version"\s*:\s*"([^"]+)"' }
    )
    foreach ($manifest in $textManifests) {
        $content = Get-Content -LiteralPath $manifest.Path -Raw
        $match = [regex]::Match($content, $manifest.Pattern)
        if (-not $match.Success -or $match.Groups[1].Value -ne $ExpectedVersion) {
            throw "Version mismatch in $($manifest.Path): expected $ExpectedVersion"
        }
    }

    $lockFiles = @(
        [pscustomobject]@{ Path = (Join-Path $ServerDir 'uv.lock'); Pattern = '(?ms)\[\[package\]\]\s*name = "diligent"\s*version = "([^"]+)"' },
        [pscustomobject]@{ Path = (Join-Path $DesktopTauriDir 'Cargo.lock'); Pattern = '(?ms)name = "diligent-desktop"\s*version = "([^"]+)"' }
    )
    foreach ($lock in $lockFiles) {
        $content = Get-Content -LiteralPath $lock.Path -Raw
        $match = [regex]::Match($content, $lock.Pattern)
        if (-not $match.Success -or $match.Groups[1].Value -ne $ExpectedVersion) {
            throw "Version mismatch in $($lock.Path): expected $ExpectedVersion"
        }
    }
    Write-Ok "Release version contract verified: $ExpectedVersion"
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
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -or [Environment]::Is64BitOperatingSystem -ne $true) {
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
    $rustHost = (& rustc -vV | Select-String '^host:').ToString()
    if ($rustHost -notmatch [regex]::Escape($DesktopTargetTriple)) {
        throw "Rust host must be $DesktopTargetTriple; found $rustHost"
    }
    $actualRustVersion = (& rustc --version).Trim()
    if ($actualRustVersion -notmatch "^rustc $([regex]::Escape($RustVersion))(?:\s|$)") {
        throw "Rust release builds require rustc $RustVersion; found $actualRustVersion"
    }
}

function Initialize-DesktopBuildDependencies {
    Install-ApplicationDependencies -BuildFrontend $false -DesktopBuild
    if (-not (Test-Path -LiteralPath $DesktopTauriCli)) {
        Invoke-Checked -FilePath $DesktopNpmCmd -WorkingDirectory $DesktopDir -ArgumentList @(
            'ci', '--ignore-scripts', '--no-audit', '--no-fund'
        )
    }
}

function Build-DesktopFrontend {
    $previousCi = $env:CI
    $env:CI = '1'
    try {
        Build-Frontend
    }
    finally {
        if ($null -eq $previousCi) { Remove-Item Env:CI -ErrorAction SilentlyContinue } else { $env:CI = $previousCi }
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
    [void](Remove-LauncherPath -Path $backendRoot -Activity 'DILIGENT: remove frozen backend staging' -Strict)
}

function Test-FrozenBackend {
    param([Parameter(Mandatory = $true)][string]$StageRoot, [Parameter(Mandatory = $true)][string]$Version)
    $backendRoot = Join-Path $StageRoot 'runtime-stage/backend'
    $backend = Join-Path $backendRoot 'DILIGENTBackend.exe'
    $dataRoot = Join-Path $StageRoot 'frozen-data'
    $readyFile = Join-Path $dataRoot 'state/ready.json'
    $qaRoot = Join-Path $RepoRoot 'assets/QA/release-audit-20260826'
    $stdoutPath = Join-Path $qaRoot 'launcher-frozen-backend.stdout.log'
    $stderrPath = Join-Path $qaRoot 'launcher-frozen-backend.stderr.log'
    New-Item -ItemType Directory -Path $qaRoot -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $readyFile), (Join-Path $dataRoot 'resources') -Force | Out-Null
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $backend
    $psi.Arguments = "--ready-file `"$readyFile`" --host 127.0.0.1"
    $psi.WorkingDirectory = $backendRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $sessionSecret = [guid]::NewGuid().ToString('N')
    foreach ($pair in @{
        DILIGENT_DESKTOP='true'; DILIGENT_DESKTOP_SESSION_SECRET=$sessionSecret; DILIGENT_RELEASE_VERSION=$Version; DILIGENT_RUNTIME_ROOT=(Join-Path $StageRoot 'runtime-stage')
        DILIGENT_DATA_ROOT=$dataRoot; DILIGENT_SQLITE_PATH=(Join-Path $dataRoot 'resources/database.db')
        DILIGENT_ACCESS_KEY_MATERIAL_FILE=(Join-Path $dataRoot 'resources/access-key-material.json'); DILIGENT_RESOURCES_PATH=(Join-Path $StageRoot 'runtime-stage/app/resources'); RELOAD='false'
    }.GetEnumerator()) { $psi.Environment[$pair.Key] = [string]$pair.Value }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $psi
    $stdoutTask = $null
    $stderrTask = $null
    try {
        if (-not $process.Start()) { throw 'Frozen backend did not start' }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $ready = $null
        for ($attempt = 0; $attempt -lt 120 -and $null -eq $ready; $attempt++) {
            if (Test-Path -LiteralPath $readyFile) { $ready = Get-Content -LiteralPath $readyFile -Raw | ConvertFrom-Json }
            elseif ($process.HasExited) { throw 'Frozen backend exited before its ready file appeared' }
            if ($null -eq $ready) { Start-Sleep -Milliseconds 500 }
        }
        if ($null -eq $ready) { throw 'Frozen backend ready-file timeout' }
        $baseUrl = "http://127.0.0.1:$($ready.port)"
        if (-not (Invoke-HealthCheck -Url "$baseUrl/api/health" -Attempts 60 -DelaySeconds 1)) {
            $process.Refresh()
            $exitState = if ($process.HasExited) { "exited with code $($process.ExitCode)" } else { 'still running' }
            throw "Frozen backend health check failed; process was $exitState"
        }
        if ((Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/" -TimeoutSec 5).StatusCode -ne 200) { throw 'Frozen backend did not serve Angular index' }
        if ((Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/clinical-sessions" -TimeoutSec 5).StatusCode -ne 200) { throw 'Frozen backend SPA fallback failed' }
        $unauthorizedStatus = 0
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/model-config" -TimeoutSec 5 | Out-Null
        }
        catch {
            $unauthorizedStatus = [int]$_.Exception.Response.StatusCode.value__
        }
        if ($unauthorizedStatus -ne 401) { throw "Frozen backend unauthorized API check returned $unauthorizedStatus" }
        $desktopSession = [Microsoft.PowerShell.Commands.WebRequestSession]::new()
        $bootstrapHeaders = @{ Origin = $baseUrl }
        Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/desktop/bootstrap" -Method Post `
            -Headers $bootstrapHeaders -ContentType 'application/json' -Body (@{ token = $sessionSecret } | ConvertTo-Json) `
            -WebSession $desktopSession -TimeoutSec 5 | Out-Null
        $authorizedConfig = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/model-config" `
            -WebSession $desktopSession -TimeoutSec 30
        if ($authorizedConfig.StatusCode -ne 200) { throw 'Frozen backend authorized model-config check failed' }
        $databasePath = Join-Path $dataRoot 'resources/database.db'
        if (-not (Test-Path -LiteralPath $databasePath -PathType Leaf)) {
            throw 'Frozen backend did not create its SQLite database during startup'
        }
        if ((Get-Item -LiteralPath $databasePath).Length -le 0) {
            throw 'Frozen backend created an empty SQLite database during startup'
        }
        Write-Ok 'Frozen backend smoke test passed, including first-run database initialization'
    }
    finally {
        if ($process -and -not $process.HasExited) { $process.Kill(); $process.WaitForExit(10000) | Out-Null }
        if ($stdoutTask) { $stdoutTask.Result | Set-Content -LiteralPath $stdoutPath -Encoding utf8 }
        if ($stderrTask) { $stderrTask.Result | Set-Content -LiteralPath $stderrPath -Encoding utf8 }
        [void](Remove-LauncherPath -Path $dataRoot -Activity 'DILIGENT: remove frozen backend test data')
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
    $frontendOutputRoot = if ($script:DesktopFrontendOutputDir) { Join-Path $script:DesktopFrontendOutputDir 'browser' } else { Join-Path $ClientDir 'dist/browser' }
    Copy-Item -Path (Join-Path $frontendOutputRoot '*') -Destination (Join-Path $payloadRoot 'app/client/dist/browser') -Recurse -Force
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
    Invoke-Checked -FilePath $VenvPython -WorkingDirectory $RepoRoot -ArgumentList @(
        (Join-Path $DesktopDir 'build/validate_runtime_archive.py'), '--archive', $runtimeOutput, '--version', $Version
    )
    $env:DILIGENT_RUNTIME_ARCHIVE = $runtimeOutput
    Write-Info "Runtime archive is ready for OUT_DIR embedding: $runtimeOutput"
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
    Invoke-Checked -FilePath $DesktopTauriCli -WorkingDirectory $DesktopDir -ArgumentList @(
        'build', '--target', $DesktopTargetTriple, '--config', $ConfigurationPath, '--no-bundle'
    )
    if ($DesktopTarget -in @('Msi', 'All')) {
        Invoke-Checked -FilePath $DesktopTauriCli -WorkingDirectory $DesktopDir -ArgumentList @(
            'bundle', '--target', $DesktopTargetTriple, '--config', $ConfigurationPath, '--bundles', 'msi'
        )
    }
}

function Test-PortableDesktopArtifact {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -lt 1MB) { throw "Portable desktop artifact is missing or unexpectedly small: $Path" }
    $stream = [IO.File]::OpenRead($Path)
    try {
        $reader = [IO.BinaryReader]::new($stream)
        if ($reader.ReadUInt16() -ne 0x5A4D) { throw 'Portable artifact is not a PE executable' }
        $stream.Seek(0x3C, [IO.SeekOrigin]::Begin) | Out-Null
        $peOffset = $reader.ReadInt32()
        $stream.Seek($peOffset, [IO.SeekOrigin]::Begin) | Out-Null
        if ($reader.ReadUInt32() -ne 0x00004550) { throw 'Portable artifact has an invalid PE signature' }
        if ($reader.ReadUInt16() -ne 0x8664) { throw 'Portable artifact is not AMD64' }
    }
    finally {
        $stream.Dispose()
    }
}

function Test-MsiMetadata {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -lt 1KB) { throw "MSI artifact is missing or unexpectedly small: $Path" }
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $null
    try {
        $database = $installer.OpenDatabase($Path, 0)
        $manufacturerView = $database.OpenView("SELECT `Value` FROM `Property` WHERE `Property`='Manufacturer'")
        $manufacturerView.Execute()
        $manufacturerRecord = $manufacturerView.Fetch()
        $manufacturer = if ($null -eq $manufacturerRecord) { '' } else { $manufacturerRecord.StringData(1) }
        $upgradeView = $database.OpenView('SELECT DISTINCT UpgradeCode FROM Upgrade')
        $upgradeView.Execute()
        $upgradeRecord = $upgradeView.Fetch()
        $upgradeCode = if ($null -eq $upgradeRecord) { '' } else { $upgradeRecord.StringData(1) }
        if ($manufacturer -ne 'CTCycle') { throw "MSI Manufacturer must be CTCycle; found $manufacturer" }
        if ($upgradeCode.Trim('{}').ToUpperInvariant() -ne '2CF8EF35-4160-59EB-89D8-01EC7D19A887') {
            throw "MSI UpgradeCode did not preserve the release identity: $upgradeCode"
        }
    }
    finally {
        if ($database) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($database) }
        if ($installer) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($installer) }
    }
}

function Copy-DesktopArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $sourceHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
        $destinationHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sourceHash -eq $destinationHash) {
            Write-Info "Keeping byte-identical existing release artifact: $Destination"
            return
        }
        throw "Refusing to overwrite a non-identical release artifact: $Destination"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination
}

function Publish-DesktopArtifacts {
    param([Parameter(Mandatory = $true)][string]$Version)
    New-Item -ItemType Directory -Path $DesktopArtifactsDir -Force | Out-Null
    $paths = Get-DesktopReleaseArtifactPaths -Version $Version
    $portable = $paths.Portable
    $msi = $paths.Msi
    $rawExe = Join-Path $CargoTargetDir "$DesktopTargetTriple/release/diligent-desktop.exe"
    if ($DesktopTarget -in @('Portable', 'All')) {
        if (-not (Test-Path -LiteralPath $rawExe -PathType Leaf)) { throw "Tauri x64 raw executable was not found: $rawExe" }
        Copy-DesktopArtifact -Source $rawExe -Destination $portable
        Test-PortableDesktopArtifact -Path $portable
    }
    if ($DesktopTarget -in @('Msi', 'All')) {
        $msiName = "DILIGENT Clinical Copilot_${Version}_x64_en-US.msi"
        $builtMsi = Join-Path $CargoTargetDir "$DesktopTargetTriple/release/bundle/msi/$msiName"
        if (-not (Test-Path -LiteralPath $builtMsi -PathType Leaf)) { throw "Tauri x64 MSI artifact was not found: $builtMsi" }
        Copy-DesktopArtifact -Source $builtMsi -Destination $msi
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
    Assert-DesktopReleaseVersion -ExpectedVersion $resolvedVersion
    $dirtyTree = Assert-CleanReleaseTree
    Assert-DesktopBuildHost
    $stageRoot = Join-Path $DesktopStageRoot "$resolvedVersion/$PID"
    New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
    $script:DesktopFrontendOutputDir = Join-Path $stageRoot 'frontend-dist'
    try {
        Initialize-DesktopBuildDependencies
        Build-DesktopFrontend
        Build-DesktopBackend -StageRoot $stageRoot
        New-DesktopRuntimeArchive -StageRoot $stageRoot -Version $resolvedVersion -DirtyTree $dirtyTree
        Test-FrozenBackend -StageRoot $stageRoot -Version $resolvedVersion
        $configuration = New-TauriReleaseConfiguration -Version $resolvedVersion
        Build-TauriApplication -ConfigurationPath $configuration -Version $resolvedVersion
        Publish-DesktopArtifacts -Version $resolvedVersion
        Write-Ok "Desktop release $resolvedVersion built under $DesktopArtifactsDir"
    }
    finally {
        if (Test-Path -LiteralPath $stageRoot) { [void](Remove-LauncherPath -Path $stageRoot -Activity 'DILIGENT: remove desktop build staging') }
        $script:DesktopFrontendOutputDir = $null
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
        if (Test-Path -LiteralPath $target) { [void](Remove-LauncherPath -Path $target -Activity "DILIGENT: remove desktop state $([IO.Path]::GetFileName($target))" -Strict) }
    }
    if (Test-Path -LiteralPath $DesktopGeneratedDir) {
        [void](Remove-LauncherPath -Path $DesktopGeneratedDir -KeepRoot -PreserveNames @('.gitkeep') -Activity 'DILIGENT: remove generated desktop state' -Strict)
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

function Get-MainMenuEntries {
    return @(
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Launch application'; Description = 'Start local services'; Key = 'Launch'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Install dependencies'; Description = 'Sync runtimes + packages'; Key = 'Install'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Rebuild frontend'; Description = 'Recreate Angular production output'; Key = 'Rebuild'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Initialize database'; Description = 'Prepare local data store'; Key = 'Database'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Run test suite'; Description = 'Execute project checks'; Key = 'Tests'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Check for updates'; Description = 'Report origin/main status only'; Key = 'Check'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Update application'; Description = 'Pull latest changes from origin/main'; Key = 'Update'; Destructive = $false }
        [pscustomobject]@{ Section = 'BUILD & DISTRIBUTION'; Label = 'Create release artifacts'; Description = 'Choose package, manifest, or all'; Key = 'CreateRelease'; Destructive = $false }
        [pscustomobject]@{ Section = 'BUILD & DISTRIBUTION'; Label = 'Remove release artifacts'; Description = 'Choose artifact(s) or all versions'; Key = 'RemoveRelease'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove logs'; Description = 'Delete application logs'; Key = 'Logs'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Clear cache'; Description = 'Remove temporary caches'; Key = 'Cache'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove all data'; Description = 'Delete local user data only'; Key = 'AllData'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Uninstall application'; Description = 'Remove generated dependencies'; Key = 'Uninstall'; Destructive = $true }
        [pscustomobject]@{ Section = 'EXIT'; Label = 'Exit'; Description = 'Close launcher'; Key = 'Exit'; Destructive = $false }
    )
}

function Write-MenuOption {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Entry,
        [Parameter(Mandatory = $true)][int]$NumberWidth,
        [Parameter(Mandatory = $true)][int]$LabelWidth
    )

    $color = if ($Entry.Destructive) { 'Yellow' } elseif ($Entry.Key -eq 'Exit') { 'DarkGray' } else { 'White' }
    $content = ("{0,$NumberWidth}.  {1,-$LabelWidth} {2}" -f $Entry.Number, $Entry.Label, $Entry.Description)
    Write-MenuLine -Text $content -Color $color
}

function Write-MenuSection([string]$Title) {
    Write-MenuLine -Text '' -Color DarkCyan
    Write-MenuLine -Text $Title -Color Yellow
}

function Show-MainMenu {
    Clear-LauncherProgress
    $entries = @(Get-MainMenuEntries)
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $entries[$index] = [pscustomobject]@{
            Number = $index + 1
            Section = $entries[$index].Section
            Label = $entries[$index].Label
            Description = $entries[$index].Description
            Key = $entries[$index].Key
            Destructive = $entries[$index].Destructive
        }
    }
    $numberWidth = ([string]$entries.Count).Length
    $labelWidth = ($entries | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
    Write-Host ''
    Write-MenuRule
    Write-MenuLine -Text 'DILIGENT  /  CLINICAL COPILOT' -Color Cyan
    Write-MenuLine -Text 'Local development and maintenance console' -Color DarkGray
    Write-MenuRule
    $lastSection = $null
    foreach ($entry in $entries) {
        if ($entry.Section -ne $lastSection) {
            Write-MenuSection -Title $entry.Section
            $lastSection = $entry.Section
        }
        Write-MenuOption -Entry $entry -NumberWidth $numberWidth -LabelWidth $labelWidth
    }
    Write-MenuRule
    return $entries
}

function Wait-ForMenu {
    Clear-LauncherProgress
    Write-Host ''
    Write-Host 'Press any key to return to the menu...' -ForegroundColor DarkGray
    if (-not $script:LauncherInteractive) { return }
    [Console]::ReadKey($true) | Out-Null
}

function Read-DesktopArtifactSelection {
    param([ValidateSet('Create', 'Remove')][string]$Operation)

    $entries = if ($Operation -eq 'Create') {
        @(
            [pscustomobject]@{ Key = 'Portable'; Label = 'Portable executable'; Description = 'Build the standalone desktop executable'; Destructive = $false }
            [pscustomobject]@{ Key = 'Msi'; Label = 'MSI installer'; Description = 'Build the Windows installer'; Destructive = $false }
            [pscustomobject]@{ Key = 'Checksum'; Label = 'SHA-256 manifest'; Description = 'Create a manifest from existing artifacts'; Destructive = $false }
            [pscustomobject]@{ Key = 'All'; Label = 'All distribution artifacts'; Description = 'Build the portable executable and MSI'; Destructive = $false }
            [pscustomobject]@{ Key = 'Back'; Label = 'Back'; Description = 'Return to the main menu'; Destructive = $false }
        )
    }
    else {
        @(
            [pscustomobject]@{ Key = 'Portable'; Label = 'Portable executable'; Description = 'Remove the selected version''s executable'; Destructive = $true }
            [pscustomobject]@{ Key = 'Msi'; Label = 'MSI installer'; Description = 'Remove the selected version''s installer'; Destructive = $true }
            [pscustomobject]@{ Key = 'Checksum'; Label = 'SHA-256 manifest'; Description = 'Remove the selected version''s manifest'; Destructive = $true }
            [pscustomobject]@{ Key = 'All'; Label = 'All artifacts for one version'; Description = 'Remove every artifact for the selected version'; Destructive = $true }
            [pscustomobject]@{ Key = 'AllVersions'; Label = 'All versions and artifacts'; Description = 'Remove every generated desktop release'; Destructive = $true }
            [pscustomobject]@{ Key = 'Back'; Label = 'Back'; Description = 'Return to the main menu'; Destructive = $false }
        )
    }
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $entries[$index] = [pscustomobject]@{
            Number = $index + 1
            Key = $entries[$index].Key
            Label = $entries[$index].Label
            Description = $entries[$index].Description
            Destructive = $entries[$index].Destructive
        }
    }
    $numberWidth = ([string]$entries.Count).Length
    $labelWidth = ($entries | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
    Write-Host ''
    Write-Host (if ($Operation -eq 'Create') { 'CREATE DESKTOP RELEASE ARTIFACTS' } else { 'REMOVE DESKTOP RELEASE ARTIFACTS' }) -ForegroundColor Yellow
    foreach ($entry in $entries) {
        Write-MenuOption -Entry $entry -NumberWidth $numberWidth -LabelWidth $labelWidth
    }
    Write-MenuRule
    $selection = (Read-Host ("Select an option (1-{0})" -f $entries.Count)).Trim()
    $selectedNumber = 0
    if (-not [int]::TryParse($selection, [ref]$selectedNumber) -or $selectedNumber -lt 1 -or $selectedNumber -gt $entries.Count) {
        throw "Invalid selection. Enter a number from 1 through $($entries.Count)."
    }
    $selectedEntry = $entries[$selectedNumber - 1]
    switch ($selectedEntry.Key) {
        'Back' { return $null }
        'AllVersions' { return [pscustomobject]@{ Target = 'All'; AllVersions = $true } }
        default { return [pscustomobject]@{ Target = $selectedEntry.Key; AllVersions = $false } }
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
    Invoke-TrackedLauncherAction -Name "action $Action" -Operation {
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
    }
    exit 0
}

while ($true) {
    $entries = @(Show-MainMenu)
    $maxOption = $entries.Count
    $rawSelection = Read-Host "Select an option (1-$maxOption)"
    if ($null -eq $rawSelection) {
        exit 0
    }
    $selection = $rawSelection.Trim()

    if ($selection -notmatch '^[1-9][0-9]*$' -or [int]$selection -lt 1 -or [int]$selection -gt $maxOption) {
        Write-Fatal "Select a number from 1 through $maxOption."
        Wait-ForMenu
        continue
    }
    $entry = $entries[[int]$selection - 1]
    if ($entry.Key -eq 'Exit') { break }

    try {
        Invoke-TrackedLauncherAction -Name $entry.Label -Operation {
            switch ($entry.Key) {
                'Launch' {
                    Start-Application
                    exit 0
                }
                'Install' { Install-OrUpdateApplication }
                'Rebuild' { Rebuild-Frontend }
                'Database' { Initialize-Database }
                'Tests' { Invoke-TestSuite }
                'Check' { Check-ForUpdates }
                'Update' { Update-Application }
                'CreateRelease' { Invoke-CreateDesktopReleaseMenu }
                'RemoveRelease' { Invoke-RemoveDesktopReleaseMenu }
                'Logs' { Remove-ApplicationLogs }
                'Cache' { Clear-ApplicationCache }
                'AllData' { Remove-AllData }
                'Uninstall' { Uninstall-Application }
            }
        }
    }
    catch {
        Write-Fatal $_.Exception.Message
        Clear-LauncherProgress
    }

    Wait-ForMenu
}
Clear-LauncherProgress

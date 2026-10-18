# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,
    [ValidateSet('Portable', 'Msi', 'All')]
    [string]$DesktopTarget = 'All',
    [switch]$InstallMsi,
    [switch]$TestUpgrade,
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$UpgradeFromVersion = '3.3.0',
    [string]$PreviousMsiPath,
    [string]$PreviousChecksumPath,
    [switch]$RequireUnsigned,
    [switch]$TestDeepPath,
    [string]$ReportDirectory
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$artifactRoot = Join-Path $repoRoot 'release'
$qaRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'assets/QA'))
if (-not $ReportDirectory) { $ReportDirectory = Join-Path $qaRoot "desktop-smoke-$Version-$([guid]::NewGuid())" }
$ReportDirectory = [IO.Path]::GetFullPath($ReportDirectory)
if (-not $ReportDirectory.StartsWith($qaRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Desktop smoke reports must be stored under assets/QA.'
}
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
$portablePath = Join-Path $artifactRoot "DILIGENT-v$Version-windows-x64-portable.exe"
$msiPath = Join-Path $artifactRoot "DILIGENT-v$Version-windows-x64.msi"
$checksumPath = Join-Path $artifactRoot "DILIGENT-v$Version-windows-x64.sha256"
$previousMsiPath = if ($PreviousMsiPath) {
    $PreviousMsiPath
}
else {
    Join-Path $artifactRoot "DILIGENT-v$UpgradeFromVersion-windows-x64.msi"
}
$previousChecksumPath = if ($PreviousChecksumPath) {
    $PreviousChecksumPath
}
else {
    Join-Path $artifactRoot "DILIGENT-v$UpgradeFromVersion-windows-x64.sha256"
}
$tempParent = if ($env:RUNNER_TEMP) {
    $env:RUNNER_TEMP
}
elseif ($env:TEMP) {
    $env:TEMP
}
else {
    [IO.Path]::GetTempPath()
}
$smokeRoot = Join-Path $tempParent ("diligent-" + [IO.Path]::GetRandomFileName())
$activeProcesses = [System.Collections.Generic.List[Diagnostics.Process]]::new()
$report = [ordered]@{
    version = $Version
    target = $DesktopTarget
    portable_artifact = [IO.Path]::GetFileName($portablePath)
    msi_artifact = [IO.Path]::GetFileName($msiPath)
}

function Assert-File([string]$path, [long]$minimumBytes) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required desktop artifact is missing: $path"
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -lt $minimumBytes) {
        throw "Desktop artifact is unexpectedly small: $path ($($item.Length) bytes)"
    }
}

function Assert-Checksum([string]$artifactPath, [string]$manifestPath) {
    Assert-File -path $artifactPath -minimumBytes 1
    Assert-File -path $manifestPath -minimumBytes 1
    $artifactName = [IO.Path]::GetFileName($artifactPath)
    $record = Get-Content -LiteralPath $manifestPath |
        Where-Object { $_ -match "^(?<hash>[0-9a-fA-F]{64})\s+\*?(?<name>.+?)\s*$" -and $Matches.name.Trim() -eq $artifactName } |
        Select-Object -First 1
    if (-not $record) {
        throw "Checksum manifest does not contain an entry for $artifactName."
    }
    $expected = [regex]::Match($record, '^(?<hash>[0-9a-fA-F]{64})').Groups['hash'].Value.ToLowerInvariant()
    $actual = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "Checksum mismatch for ${artifactName}: expected $expected, found $actual."
    }
    return $actual
}

function Assert-PortablePe([string]$path) {
    $stream = [IO.File]::OpenRead($path)
    try {
        $reader = [IO.BinaryReader]::new($stream)
        if ($reader.ReadUInt16() -ne 0x5A4D) { throw 'Portable artifact is not a PE executable.' }
        $stream.Seek(0x3C, [IO.SeekOrigin]::Begin) | Out-Null
        $peOffset = $reader.ReadInt32()
        $stream.Seek($peOffset, [IO.SeekOrigin]::Begin) | Out-Null
        if ($reader.ReadUInt32() -ne 0x00004550) { throw 'Portable artifact has an invalid PE signature.' }
        if ($reader.ReadUInt16() -ne 0x8664) { throw 'Portable artifact is not an AMD64 executable.' }
    }
    finally {
        $stream.Dispose()
    }
}

function Get-MsiProperty($database, [string]$name) {
    $view = $null
    $record = $null
    try {
        $view = $database.OpenView("SELECT `Value` FROM `Property` WHERE `Property`='$name'")
        [void]$view.Execute()
        $record = $view.Fetch()
        if ($null -eq $record) { return '' }
        $value = [string]$record.StringData(1)
        return $value
    }
    finally {
        if ($record) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($record) }
        if ($view) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($view) }
    }
}

function Get-MsiMetadata([string]$path) {
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $null
    $upgradeView = $null
    $upgradeRecord = $null
    try {
        $database = $installer.OpenDatabase($path, 0)
        $upgradeView = $database.OpenView('SELECT DISTINCT UpgradeCode FROM Upgrade')
        $upgradeView.Execute()
        $upgradeRecord = $upgradeView.Fetch()
        $upgradeCode = if ($null -eq $upgradeRecord) { '' } else { [string]$upgradeRecord.StringData(1) }
        return [pscustomobject]@{
            ProductCode = Get-MsiProperty $database 'ProductCode'
            ProductName = Get-MsiProperty $database 'ProductName'
            ProductVersion = Get-MsiProperty $database 'ProductVersion'
            Manufacturer = Get-MsiProperty $database 'Manufacturer'
            UpgradeCode = $upgradeCode
        }
    }
    finally {
        if ($upgradeRecord) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($upgradeRecord) }
        if ($upgradeView) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($upgradeView) }
        if ($database) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($database) }
        if ($installer) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($installer) }
    }
}

function Wait-Http([string]$uri, [int]$timeoutSeconds = 60) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    $lastError = $null
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $uri -TimeoutSec 5
            if ($response.StatusCode -eq 200) { return $response }
            $lastError = "HTTP $($response.StatusCode)"
        }
        catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Timed out waiting for $uri. Last error: $lastError"
}

function Wait-HttpStatus([string]$uri, [int]$expectedStatus, [int]$timeoutSeconds = 60) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    $lastStatus = $null
    $lastError = $null
    while ((Get-Date) -lt $deadline) {
        try {
            # Windows PowerShell 5.1 does not expose -SkipHttpErrorCheck. A
            # non-2xx response is surfaced through the exception response in
            # that host, while PowerShell 7 can return it normally.
            $invokeArguments = @{
                UseBasicParsing = $true
                Uri = $uri
                TimeoutSec = 5
            }
            if ((Get-Command Invoke-WebRequest).Parameters.ContainsKey('SkipHttpErrorCheck')) {
                $invokeArguments.SkipHttpErrorCheck = $true
            }
            $response = Invoke-WebRequest @invokeArguments
            $lastStatus = [int]$response.StatusCode
            if ($lastStatus -eq $expectedStatus) { return $response }
        }
        catch {
            $errorResponse = $_.Exception.Response
            if ($null -ne $errorResponse) {
                try {
                    $lastStatus = [int]$errorResponse.StatusCode
                    if ($lastStatus -eq $expectedStatus) {
                        return [pscustomobject]@{ StatusCode = $lastStatus }
                    }
                }
                catch {
                }
            }
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    $observed = if ($null -ne $lastStatus) { "HTTP $lastStatus" } else { $lastError }
    throw "Timed out waiting for $uri to return HTTP $expectedStatus. Last result: $observed"
}

function Wait-ReadyPayload(
    [string]$readyPath,
    [string]$expectedVersion,
    [datetime]$previousWriteTimeUtc,
    [int]$previousPid,
    [int]$timeoutSeconds = 180
) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $readyPath -PathType Leaf) {
            try {
                $file = Get-Item -LiteralPath $readyPath
                $ready = Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json
                $changed = $previousWriteTimeUtc -eq [datetime]::MinValue -or
                    $file.LastWriteTimeUtc -gt $previousWriteTimeUtc -or
                    [int]$ready.pid -ne $previousPid
                if ($changed -and
                    [string]$ready.release_version -eq $expectedVersion -and
                    [int]$ready.port -gt 0 -and
                    [int]$ready.pid -gt 0) {
                    return [pscustomobject]@{
                        Ready = $ready
                        LastWriteTimeUtc = $file.LastWriteTimeUtc
                    }
                }
            }
            catch {
                # The backend can replace the JSON atomically while it starts.
            }
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Timed out waiting for a fresh ready payload at $readyPath."
}

function Assert-PackagedApiSurface([string]$baseUri) {
    $publicPaths = @(
        '/',
        '/api/health'
    )
    foreach ($path in $publicPaths) {
        $response = Wait-Http ($baseUri.TrimEnd('/') + $path)
        if ($response.StatusCode -ne 200) {
            throw "Packaged API $path returned HTTP $($response.StatusCode), expected 200."
        }
    }
    $desktopProtectedPaths = @(
        '/api/settings',
        '/api/model-config',
        '/api/inspection/sessions',
        '/api/inspection/rag/documents',
        '/api/inspection/rag/vector-store'
    )
    foreach ($path in $desktopProtectedPaths) {
        $response = Wait-HttpStatus ($baseUri.TrimEnd('/') + $path) 401
        if ($response.StatusCode -ne 401) {
            throw "Packaged API $path returned HTTP $($response.StatusCode), expected the unauthenticated desktop boundary HTTP 401."
        }
    }
}

function Wait-ProcessGone([int]$processId, [int]$timeoutSeconds = 20) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -eq (Get-Process -Id $processId -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Test-PortClosed([int]$port) {
    return $null -eq (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
}

function Start-DesktopProcess([string]$executable, [string]$isolatedRoot) {
    $tempRoot = Join-Path $isolatedRoot 'Temp'
    $roamingRoot = Join-Path $isolatedRoot 'AppData/Roaming'
    New-Item -ItemType Directory -Path $tempRoot, $roamingRoot -Force | Out-Null
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $executable
    $startInfo.WorkingDirectory = Split-Path -Parent $executable
    $startInfo.UseShellExecute = $false
    $startInfo.Environment['LOCALAPPDATA'] = $isolatedRoot
    $startInfo.Environment['TEMP'] = $tempRoot
    $startInfo.Environment['TMP'] = $tempRoot
    $startInfo.Environment['APPDATA'] = $roamingRoot
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { throw "Desktop executable did not start: $executable" }
    $activeProcesses.Add($process)
    return $process
}

function Stop-DesktopProcess([Diagnostics.Process]$process, [int]$backendPid, [int]$port) {
    if ($null -eq $process) { return }
    $process.Refresh()
    if (-not $process.HasExited) {
        if (-not $process.CloseMainWindow()) {
            throw "Desktop executable did not accept a normal close request (PID $($process.Id))."
        }
        $process.WaitForExit(30000) | Out-Null
        if (-not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit(10000) | Out-Null
            throw "Desktop executable did not exit after the normal close request (PID $($process.Id))."
        }
    }
    if (-not (Wait-ProcessGone -processId $backendPid) -or -not (Test-PortClosed -port $port)) {
        throw "Desktop backend did not stop cleanly (PID $backendPid, port $port)."
    }
}

function Invoke-PortableSmoke(
    [string]$executable,
    [string]$label,
    [string]$expectedVersion = $Version,
    [string]$isolatedRoot
) {
    if ($RequireUnsigned -and $expectedVersion -eq $Version -and
        (Get-AuthenticodeSignature -LiteralPath $executable).Status -ne 'NotSigned') {
        throw 'The candidate must match the explicitly unsigned release policy.'
    }
    if ([string]::IsNullOrWhiteSpace($isolatedRoot)) {
        $isolatedRoot = Join-Path $smokeRoot $label
    }
    $readyPath = Join-Path $isolatedRoot 'DILIGENT/data/state/desktop-backend-ready.json'
    $process = $null
    $launches = [System.Collections.Generic.List[object]]::new()
    $extractionMarker = $null
    $secondLaunchReplacedReadyFile = $false
    try {
        New-Item -ItemType Directory -Path $isolatedRoot -Force | Out-Null
        for ($launchIndex = 1; $launchIndex -le 2; $launchIndex++) {
            $previousWriteTimeUtc = [datetime]::MinValue
            $previousPid = 0
            if (Test-Path -LiteralPath $readyPath -PathType Leaf) {
                $previousWriteTimeUtc = (Get-Item -LiteralPath $readyPath).LastWriteTimeUtc
                try {
                    $previousPid = [int](Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json).pid
                }
                catch {
                    $previousPid = 0
                }
            }
            $process = Start-DesktopProcess -executable $executable -isolatedRoot $isolatedRoot
            $readyResult = Wait-ReadyPayload `
                -readyPath $readyPath `
                -expectedVersion $expectedVersion `
                -previousWriteTimeUtc $previousWriteTimeUtc `
                -previousPid $previousPid
            $ready = $readyResult.Ready
            $port = [int]$ready.port
            $backendPid = [int]$ready.pid
            Wait-Http "http://127.0.0.1:$port/api/health" | Out-Null
            Assert-PackagedApiSurface -baseUri "http://127.0.0.1:$port"
            $windowTitle = ''
            $titleDeadline = (Get-Date).AddSeconds(30)
            while ((Get-Date) -lt $titleDeadline) {
                $process.Refresh()
                $windowTitle = $process.MainWindowTitle
                if ($windowTitle -eq 'DILIGENT Clinical Copilot') { break }
                Start-Sleep -Milliseconds 250
            }
            if ($windowTitle -ne 'DILIGENT Clinical Copilot') {
                throw "$label window title was '$windowTitle', expected 'DILIGENT Clinical Copilot'."
            }
            if ($launchIndex -eq 1) {
                $runtimeCache = if ($expectedVersion -eq $UpgradeFromVersion -and $expectedVersion -ne $Version) { 'runtime' } else { 'rt' }
                $runtimeRoot = Join-Path $isolatedRoot "DILIGENT/$runtimeCache/$expectedVersion"
                $extractionMarker = Get-ChildItem -LiteralPath $runtimeRoot -Filter 'extraction.complete' -File -Recurse -ErrorAction SilentlyContinue |
                    Select-Object -First 1
                if ($null -eq $extractionMarker) {
                    throw "$label runtime extraction marker was not found under $runtimeRoot."
                }
            }
            Stop-DesktopProcess -process $process -backendPid $backendPid -port $port
            $launches.Add([ordered]@{
                launch = $launchIndex
                health = 200
                ready_release_version = [string]$ready.release_version
                ready_pid = $backendPid
                ready_port = $port
                window_title = $windowTitle
                backend_stopped = $true
                port_closed = $true
            })
            if ($launchIndex -eq 2) {
                $secondLaunchReplacedReadyFile = $previousPid -ne $backendPid -or
                    $readyResult.LastWriteTimeUtc -gt $previousWriteTimeUtc
            }
            $process.Dispose()
            $process = $null
        }
        if (-not $secondLaunchReplacedReadyFile) {
            throw "$label second launch did not replace the stale ready-file payload."
        }
        $report[$label] = [ordered]@{
            launches = @($launches)
            extraction_complete = $true
            extraction_marker = $extractionMarker.FullName
            second_launch_replaced_stale_ready_file = $secondLaunchReplacedReadyFile
        }
        return $isolatedRoot
    }
    finally {
        if ($null -ne $process) {
            $process.Refresh()
            if (-not $process.HasExited) {
                try { $process.Kill(); $process.WaitForExit(10000) | Out-Null } catch { }
            }
            $process.Dispose()
        }
    }
}

function Find-InstalledProduct([string]$productCode, [string]$productName) {
    $roots = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )
    Get-ItemProperty -Path $roots -ErrorAction SilentlyContinue |
        Where-Object {
            $_.PSChildName -eq $productCode -or
            ([string]$_.DisplayName).Trim() -eq $productName
        } |
        Select-Object -First 1
}

function Resolve-InstalledExecutable($entry) {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($entry.InstallLocation) {
        $candidates.Add((Join-Path ([string]$entry.InstallLocation) 'diligent-desktop.exe'))
    }
    if ($entry.DisplayIcon) {
        $icon = ([string]$entry.DisplayIcon).Trim().Trim('"')
        if ($icon) { $candidates.Add($icon) }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Invoke-MsiOperation([string]$operation, [string]$package, [string]$label) {
    $log = Join-Path $ReportDirectory "$label.log"
    $result = Start-Process -FilePath 'msiexec.exe' -ArgumentList @(
        $operation, ('"{0}"' -f $package), '/qn', '/norestart', '/L*v', ('"{0}"' -f $log)
    ) -Wait -PassThru -WindowStyle Hidden
    if ($result.ExitCode -notin @(0, 3010)) { throw "$label failed with MSI exit code $($result.ExitCode); see $log" }
    $report[$label] = @{ exit_code = $result.ExitCode; log = Split-Path -Leaf $log }
}

function Invoke-UpgradeFixture([string]$upgradeRoot, [switch]$Seed, [switch]$Capture, [switch]$CheckHead) {
    $dataRoot = Join-Path $upgradeRoot 'DILIGENT/data'
    $database = Join-Path $dataRoot 'resources/database.db'
    $python = Join-Path $repoRoot 'app/server/.venv/Scripts/python.exe'
    if ($Seed) {
        $archive = Join-Path $smokeRoot 'baseline-source.zip'
        & git -C $repoRoot archive --format=zip "--output=$archive" "v$UpgradeFromVersion" app/server
        if ($LASTEXITCODE -ne 0) { throw 'Unable to archive the published baseline source.' }
        $baselineSource = Join-Path $smokeRoot 'baseline-source'
        Expand-Archive -LiteralPath $archive -DestinationPath $baselineSource
        $sourceRoot = Join-Path $baselineSource 'app/server'
        $baselineMarker = Get-ChildItem -LiteralPath (Join-Path $upgradeRoot "DILIGENT/runtime/$UpgradeFromVersion") -Filter extraction.complete -File -Recurse | Select-Object -First 1
        if ($null -eq $baselineMarker) { throw 'The baseline application runtime was not extracted.' }
        $environment = @{
            DILIGENT_SQLITE_PATH = $database; DATABASE_SQLITE_PATH = $database
            DILIGENT_DATA_ROOT = $dataRoot; DATABASE_BACKEND = 'sqlite'; EMBEDDED_DATABASE = 'true'
            DILIGENT_RUNTIME_ROOT = $baselineMarker.DirectoryName
            DILIGENT_RELEASE_VERSION = $UpgradeFromVersion
            PYTHONHOME = Join-Path $repoRoot 'runtimes/python'
        }
        $saved = @{}
        try {
            foreach ($key in $environment.Keys) {
                $saved[$key] = [Environment]::GetEnvironmentVariable($key)
                [Environment]::SetEnvironmentVariable($key, $environment[$key])
            }
            & $python (Join-Path $PSScriptRoot 'seed_upgrade_fixture.py') $sourceRoot
            if ($LASTEXITCODE -ne 0) { throw 'Baseline application repository seeding failed.' }
        }
        finally {
            foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key]) }
        }
    }
    $arguments = @((Join-Path $PSScriptRoot 'check_upgrade_fixture.py'), $database, (Join-Path $ReportDirectory 'upgrade-data.json'))
    if ($Capture) { $arguments += '--capture' }
    if ($CheckHead) { $arguments += @('--expected-head', '202609170001') }
    $previousPythonHome = $env:PYTHONHOME
    try {
        $env:PYTHONHOME = Join-Path $repoRoot 'runtimes/python'
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw 'MSI database preservation/integrity check failed.' }
    }
    finally { $env:PYTHONHOME = $previousPythonHome }
}

function Invoke-MsiSmoke {
    if (-not $InstallMsi) {
        throw 'MSI smoke requires the explicit -InstallMsi switch.'
    }
    $metadata = Get-MsiMetadata -path $msiPath
    $administrator = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $administrator) { throw 'MSI lifecycle smoke requires an administrator-capable disposable host.' }
    if ($null -ne (Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName)) {
        throw 'Refusing to alter an existing DILIGENT installation; use a clean disposable host.'
    }
    if ($RequireUnsigned -and (Get-AuthenticodeSignature -LiteralPath $msiPath).Status -ne 'NotSigned') {
        throw 'The MSI must match the explicitly unsigned release policy.'
    }
    if ($metadata.ProductCode -notmatch '^\{[0-9A-Fa-f-]+\}$') { throw "MSI ProductCode is invalid: $($metadata.ProductCode)" }
    if ($metadata.ProductVersion -ne $Version) { throw "MSI ProductVersion $($metadata.ProductVersion) does not match $Version" }
    if ($metadata.Manufacturer -ne 'CTCycle') { throw "MSI Manufacturer must be CTCycle, found $($metadata.Manufacturer)" }
    if ($metadata.UpgradeCode.Trim('{}').ToUpperInvariant() -ne '2CF8EF35-4160-59EB-89D8-01EC7D19A887') {
        throw "MSI UpgradeCode did not preserve the release identity: $($metadata.UpgradeCode)"
    }
    $report.msi_metadata = [ordered]@{
        product_code = $metadata.ProductCode
        product_name = $metadata.ProductName
        product_version = $metadata.ProductVersion
        manufacturer = $metadata.Manufacturer
        upgrade_code = $metadata.UpgradeCode
    }
    $installed = $false
    $uninstallAttempted = $false
    try {
        Invoke-MsiOperation '/i' $msiPath 'msi_install'
        $installed = $true
        $entry = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -eq $entry) { throw 'Installed MSI product was not found in the Windows uninstall registry.' }
        $installedExecutable = Resolve-InstalledExecutable $entry
        if (-not $installedExecutable) { throw 'Installed MSI executable was not found.' }
        $installed = $true
        Invoke-PortableSmoke -executable $installedExecutable -label 'msi_launch'
        Invoke-MsiOperation '/x' $metadata.ProductCode 'msi_uninstall'
        $installed = $false
        $uninstallAttempted = $true
        $remaining = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -ne $remaining -or (Test-Path -LiteralPath $installedExecutable -PathType Leaf)) {
            throw 'MSI uninstall left the product registered or executable on disk.'
        }
        $report.msi_install = [ordered]@{
            installed = $true
            launched = $true
            uninstalled = $true
            product_absent = $true
        }
        Invoke-MsiOperation '/i' $msiPath 'msi_reinstall'
        $installed = $true
        $uninstallAttempted = $false
        $reinstalled = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -eq $reinstalled) { throw 'Reinstalled MSI product was not found.' }
        $installedExecutable = Resolve-InstalledExecutable $reinstalled
        if (-not $installedExecutable) { throw 'Reinstalled executable was not found.' }
        Invoke-PortableSmoke -executable $installedExecutable -label 'msi_reinstall_launch' | Out-Null
        Invoke-MsiOperation '/x' $metadata.ProductCode 'msi_reinstall_uninstall'
        $installed = $false
        if ($null -ne (Find-InstalledProduct $metadata.ProductCode $metadata.ProductName) -or
            (Test-Path -LiteralPath $installedExecutable)) { throw 'Reinstall uninstall left product files or registration.' }
        $report.msi_reinstall = @{ installed = $true; launched = $true; uninstalled = $true; product_absent = $true }
    }
    finally {
        if ($installed -and -not $uninstallAttempted) {
            try {
                Start-Process -FilePath 'msiexec.exe' -ArgumentList '/x', $metadata.ProductCode, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden | Out-Null
            }
            catch {
            }
        }
    }
}

function Invoke-MsiUpgradeSmoke {
    if (-not $TestUpgrade) { return }
    if (-not $InstallMsi) {
        throw 'MSI upgrade smoke requires the explicit -InstallMsi switch.'
    }
    if ($DesktopTarget -notin @('Msi', 'All')) {
        throw 'MSI upgrade smoke requires -DesktopTarget Msi or All.'
    }
    $metadata = Get-MsiMetadata -path $msiPath
    $previousMsiPath = (Resolve-Path -LiteralPath $previousMsiPath).Path
    $previousChecksumPath = (Resolve-Path -LiteralPath $previousChecksumPath).Path
    $previousHash = Assert-Checksum -artifactPath $previousMsiPath -manifestPath $previousChecksumPath
    $previousMetadata = Get-MsiMetadata -path $previousMsiPath
    if ($previousMetadata.ProductVersion -ne $UpgradeFromVersion) {
        throw "Previous MSI ProductVersion $($previousMetadata.ProductVersion) does not match $UpgradeFromVersion."
    }
    if ($previousMetadata.UpgradeCode.Trim('{}').ToUpperInvariant() -ne '2CF8EF35-4160-59EB-89D8-01EC7D19A887') {
        throw "Previous MSI UpgradeCode did not preserve the release identity: $($previousMetadata.UpgradeCode)"
    }
    $upgradeRoot = Join-Path $smokeRoot 'msi_upgrade'
    New-Item -ItemType Directory -Path $upgradeRoot -Force | Out-Null
    $installedPrevious = $false
    $uninstallAttempted = $false
    $installedExecutable = $null
    $markerPath = Join-Path $upgradeRoot 'DILIGENT/data/state/upgrade-survival-marker.txt'
    try {
        Invoke-MsiOperation '/i' $previousMsiPath 'msi_baseline_install'
        $installedPrevious = $true
        $previousEntry = Find-InstalledProduct -productCode $previousMetadata.ProductCode -productName $previousMetadata.ProductName
        if ($null -eq $previousEntry) { throw 'Previous MSI product was not found after installation.' }
        $previousExecutable = Resolve-InstalledExecutable $previousEntry
        if (-not $previousExecutable) { throw 'Previous MSI executable was not found.' }
        $installedPrevious = $true
        Invoke-PortableSmoke -executable $previousExecutable -label 'msi_upgrade_previous' -expectedVersion $UpgradeFromVersion -isolatedRoot $upgradeRoot | Out-Null
        Invoke-UpgradeFixture -upgradeRoot $upgradeRoot -Seed -Capture
        Invoke-PortableSmoke -executable $previousExecutable -label 'msi_upgrade_populated_previous' -expectedVersion $UpgradeFromVersion -isolatedRoot $upgradeRoot | Out-Null
        Invoke-UpgradeFixture -upgradeRoot $upgradeRoot
        New-Item -ItemType Directory -Path (Split-Path -Parent $markerPath) -Force | Out-Null
        Set-Content -LiteralPath $markerPath -Value 'persistent user-data marker' -Encoding UTF8

        Invoke-MsiOperation '/i' $msiPath 'msi_upgrade_install'
        $currentEntry = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -eq $currentEntry) { throw 'Current MSI product was not found after upgrade.' }
        if ([string]$currentEntry.DisplayVersion -ne $Version) {
            throw "Current MSI metadata does not report version $Version after upgrade."
        }
        $installedExecutable = Resolve-InstalledExecutable $currentEntry
        if (-not $installedExecutable) { throw 'Upgraded MSI executable was not found.' }
        Invoke-PortableSmoke -executable $installedExecutable -label 'msi_upgrade_current' -expectedVersion $Version -isolatedRoot $upgradeRoot | Out-Null
        Invoke-UpgradeFixture -upgradeRoot $upgradeRoot -CheckHead
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw 'Persistent user-data marker did not survive the MSI upgrade.'
        }
        Invoke-MsiOperation '/x' $metadata.ProductCode 'msi_upgrade_uninstall'
        $uninstallAttempted = $true
        $remaining = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -ne $remaining -or ($installedExecutable -and (Test-Path -LiteralPath $installedExecutable -PathType Leaf))) {
            throw 'MSI upgrade uninstall left the product registered or executable on disk.'
        }
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw 'Persistent user data was removed by MSI uninstall.'
        }
        Invoke-UpgradeFixture -upgradeRoot $upgradeRoot -CheckHead
        $report.msi_upgrade = [ordered]@{
            previous_version = $UpgradeFromVersion
            previous_msi_sha256 = $previousHash
            upgraded_version = $Version
            launched_previous = $true
            marker_survived_upgrade = $true
            launched_current = $true
            uninstalled_current = $true
            binaries_and_registration_removed = $true
            user_data_survived_uninstall = $true
            sessions_versions_settings_preserved = $true
            database_integrity_verified = $true
            migration_head_verified = $true
        }
    }
    finally {
        if ($installedPrevious -and -not $uninstallAttempted) {
            try {
                Start-Process -FilePath 'msiexec.exe' -ArgumentList '/x', $metadata.ProductCode, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden | Out-Null
            }
            catch {
            }
            try {
                Start-Process -FilePath 'msiexec.exe' -ArgumentList '/x', $previousMetadata.ProductCode, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden | Out-Null
            }
            catch {
            }
        }
    }
}

try {
    New-Item -ItemType Directory -Path $smokeRoot -Force | Out-Null
    if ($DesktopTarget -in @('Portable', 'All')) {
        Assert-File -path $portablePath -minimumBytes 1MB
        Assert-PortablePe -path $portablePath
        $portableHash = Assert-Checksum -artifactPath $portablePath -manifestPath $checksumPath
        Invoke-PortableSmoke -executable $portablePath -label 'portable_launch'
        if ($TestDeepPath) {
            $padding = 107 - $smokeRoot.Length - 1
            if ($padding -lt 1) { throw 'Set TEMP to a shorter path for the exact deep-path replay.' }
            $deepRoot = Join-Path $smokeRoot ('d' * $padding)
            $deepDirectory = Join-Path $deepRoot ('e' * (230 - $deepRoot.Length - 2 - 'portable.exe'.Length))
            New-Item -ItemType Directory -Path $deepDirectory -Force | Out-Null
            $deepExecutable = Join-Path $deepDirectory 'portable.exe'
            Copy-Item -LiteralPath $portablePath -Destination $deepExecutable
            Invoke-PortableSmoke -executable $deepExecutable -label 'portable_deep_path' -isolatedRoot $deepRoot | Out-Null
            $report.deep_path = @{ executable_length = $deepExecutable.Length; localappdata_length = $deepRoot.Length }
        }
        $report.portable_sha256 = $portableHash
    }
    if ($DesktopTarget -in @('Msi', 'All')) {
        Assert-File -path $msiPath -minimumBytes 1KB
        $msiHash = Assert-Checksum -artifactPath $msiPath -manifestPath $checksumPath
        Invoke-MsiSmoke
        Invoke-MsiUpgradeSmoke
        $report.msi_sha256 = $msiHash
    }
    $report.success = $true
}
catch {
    $report.success = $false
    $report.error = $_.Exception.Message
    throw
}
finally {
    foreach ($process in $activeProcesses) {
        try {
            $process.Refresh()
            if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit(10000) | Out-Null }
        }
        catch {
        }
        $process.Dispose()
    }
    $resolvedSmokeRoot = [IO.Path]::GetFullPath($smokeRoot)
    $resolvedTempParent = [IO.Path]::GetFullPath($tempParent).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedSmokeRoot.StartsWith($resolvedTempParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing smoke cleanup outside the temporary root.'
    }
    $cleanupError = $null
    if (Test-Path -LiteralPath $smokeRoot) {
        try { Remove-Item -LiteralPath $smokeRoot -Recurse -Force -ErrorAction Stop }
        catch {
            $cleanupError = $_.Exception.Message
            $report.success = $false
            $report.cleanup_error = $cleanupError
        }
    }
    $json = $report | ConvertTo-Json -Depth 8
    Set-Content -LiteralPath (Join-Path $ReportDirectory 'smoke.json') -Value $json -Encoding utf8
    $json
    if ($cleanupError) { throw "Desktop smoke cleanup failed: $cleanupError" }
}

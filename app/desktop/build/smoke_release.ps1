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
    [string]$PreviousChecksumPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$artifactRoot = Join-Path $repoRoot 'release'
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
            $response = Invoke-WebRequest -UseBasicParsing -SkipHttpErrorCheck -Uri $uri -TimeoutSec 5
            $lastStatus = [int]$response.StatusCode
            if ($lastStatus -eq $expectedStatus) { return $response }
        }
        catch {
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
                $runtimeRoot = Join-Path $isolatedRoot "DILIGENT/runtime/$expectedVersion"
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

function Invoke-MsiSmoke {
    if (-not $InstallMsi) {
        throw 'MSI smoke requires the explicit -InstallMsi switch.'
    }
    $metadata = Get-MsiMetadata -path $msiPath
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
        $install = Start-Process -FilePath 'msiexec.exe' -ArgumentList '/i', $msiPath, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden
        if ($install.ExitCode -notin @(0, 3010)) { throw "MSI install failed with exit code $($install.ExitCode)." }
        $entry = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -eq $entry) { throw 'Installed MSI product was not found in the Windows uninstall registry.' }
        $installedExecutable = Resolve-InstalledExecutable $entry
        if (-not $installedExecutable) { throw 'Installed MSI executable was not found.' }
        $installed = $true
        Invoke-PortableSmoke -executable $installedExecutable -label 'msi_launch'
        $uninstall = Start-Process -FilePath 'msiexec.exe' -ArgumentList '/x', $metadata.ProductCode, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden
        $uninstallAttempted = $true
        if ($uninstall.ExitCode -notin @(0, 3010)) { throw "MSI uninstall failed with exit code $($uninstall.ExitCode)." }
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
        $installPrevious = Start-Process -FilePath 'msiexec.exe' -ArgumentList '/i', $previousMsiPath, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden
        if ($installPrevious.ExitCode -notin @(0, 3010)) {
            throw "Previous MSI install failed with exit code $($installPrevious.ExitCode)."
        }
        $previousEntry = Find-InstalledProduct -productCode $previousMetadata.ProductCode -productName $previousMetadata.ProductName
        if ($null -eq $previousEntry) { throw 'Previous MSI product was not found after installation.' }
        $previousExecutable = Resolve-InstalledExecutable $previousEntry
        if (-not $previousExecutable) { throw 'Previous MSI executable was not found.' }
        $installedPrevious = $true
        Invoke-PortableSmoke -executable $previousExecutable -label 'msi_upgrade_previous' -expectedVersion $UpgradeFromVersion -isolatedRoot $upgradeRoot | Out-Null
        New-Item -ItemType Directory -Path (Split-Path -Parent $markerPath) -Force | Out-Null
        Set-Content -LiteralPath $markerPath -Value 'persistent user-data marker' -Encoding UTF8

        $upgrade = Start-Process -FilePath 'msiexec.exe' -ArgumentList '/i', $msiPath, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden
        if ($upgrade.ExitCode -notin @(0, 3010)) {
            throw "MSI upgrade to $Version failed with exit code $($upgrade.ExitCode)."
        }
        $currentEntry = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -eq $currentEntry) { throw 'Current MSI product was not found after upgrade.' }
        if ([string]$currentEntry.DisplayVersion -ne $Version) {
            throw "Current MSI metadata does not report version $Version after upgrade."
        }
        $installedExecutable = Resolve-InstalledExecutable $currentEntry
        if (-not $installedExecutable) { throw 'Upgraded MSI executable was not found.' }
        Invoke-PortableSmoke -executable $installedExecutable -label 'msi_upgrade_current' -expectedVersion $Version -isolatedRoot $upgradeRoot | Out-Null
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw 'Persistent user-data marker did not survive the MSI upgrade.'
        }
        $uninstall = Start-Process -FilePath 'msiexec.exe' -ArgumentList '/x', $metadata.ProductCode, '/qn', '/norestart' -Wait -PassThru -WindowStyle Hidden
        $uninstallAttempted = $true
        if ($uninstall.ExitCode -notin @(0, 3010)) { throw "MSI upgrade uninstall failed with exit code $($uninstall.ExitCode)." }
        $remaining = Find-InstalledProduct -productCode $metadata.ProductCode -productName $metadata.ProductName
        if ($null -ne $remaining -or ($installedExecutable -and (Test-Path -LiteralPath $installedExecutable -PathType Leaf))) {
            throw 'MSI upgrade uninstall left the product registered or executable on disk.'
        }
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw 'Persistent user data was removed by MSI uninstall.'
        }
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
    if (Test-Path -LiteralPath $smokeRoot) {
        Remove-Item -LiteralPath $smokeRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    $report | ConvertTo-Json -Depth 8
}

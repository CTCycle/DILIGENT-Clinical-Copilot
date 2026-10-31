$ErrorActionPreference = 'Stop'
$modulePath = Join-Path $PSScriptRoot '..\scripts\launcher_cleanup.psm1'
Import-Module -Name $modulePath -Force

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Write-FixtureFile {
    param([string]$Path, [string]$Content = 'fixture')
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [IO.File]::WriteAllText($Path, $Content)
}

$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) "diligent-cleanup-tests-$([guid]::NewGuid().ToString('N'))"
$repoRoot = Join-Path $fixtureRoot 'repository'
$localAppData = Join-Path $fixtureRoot 'profile\AppData\Local'
$appRoot = Join-Path $localAppData 'DILIGENT'
$dataRoot = Join-Path $appRoot 'data'
$outsideRoot = Join-Path $fixtureRoot 'outside'
$junctionPath = Join-Path $repoRoot 'app\server\linked-cache'
$lockedStream = $null
$aclTarget = Join-Path $fixtureRoot 'acl-denied'
$aclSnapshot = $null

try {
    foreach ($path in @($repoRoot, $dataRoot, $outsideRoot)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }

    $disposableFiles = @(
        (Join-Path $repoRoot 'runtimes\cache\angular\out-tsc\spec\case.js'),
        (Join-Path $repoRoot 'runtimes\cache\logs\source.log'),
        (Join-Path $repoRoot 'app\server\pkg\__pycache__\module.pyc'),
        (Join-Path $repoRoot 'app\server\.pytest_cache\v\cache\lastfailed'),
        (Join-Path $repoRoot 'app\server\.venv\Lib\site-packages\pkg\__pycache__\module.pyc'),
        (Join-Path $repoRoot 'app\client\.angular\cache\angular.bin'),
        (Join-Path $repoRoot 'app\client\node_modules\.cache\bundler.bin'),
        (Join-Path $repoRoot 'app\client\node_modules\.vite\deps\metadata.json'),
        (Join-Path $repoRoot 'app\client\dist\browser\index.html'),
        (Join-Path $repoRoot 'app\desktop\node_modules\.cache\tauri.bin'),
        (Join-Path $repoRoot 'app\desktop\src-tauri\gen\schemas\schema.json'),
        (Join-Path $repoRoot 'app\desktop\src-tauri\generated\generated.json'),
        (Join-Path $repoRoot 'app\tests\unit\__pycache__\contract.pyc'),
        (Join-Path $repoRoot 'app\tests\assets\QA\old-run\pytest-cache\cache.json'),
        (Join-Path $repoRoot 'assets\QA\desktop-release-staging\staging\partial.bin'),
        (Join-Path $repoRoot '.coverage.worker-123'),
        (Join-Path $dataRoot 'cache\embeddings\model.bin'),
        (Join-Path $dataRoot 'cache\logs\desktop-backend.log'),
        (Join-Path $dataRoot 'settings\..env.1234.tmp'),
        (Join-Path $dataRoot '.database_config.json.abcdef.tmp'),
        (Join-Path $appRoot 'runtime\3.5.1\.extract-0123456789abcdef0123456789abcdef\partial.bin')
    )
    foreach ($path in $disposableFiles) { Write-FixtureFile -Path $path }
    Write-FixtureFile -Path (Join-Path $repoRoot 'runtimes\cache\.gitkeep')

    $persistentFiles = @(
        (Join-Path $dataRoot 'database.db'),
        (Join-Path $dataRoot 'database_config.json'),
        (Join-Path $dataRoot 'sources\documents\patient.pdf'),
        (Join-Path $dataRoot 'sources\vectors\index.bin'),
        (Join-Path $dataRoot 'exports\report.json'),
        (Join-Path $dataRoot 'settings\.env'),
        (Join-Path $dataRoot 'access-key-material.json'),
        (Join-Path $repoRoot 'settings\.env'),
        (Join-Path $repoRoot 'app\client\package-lock.json')
    )
    foreach ($path in $persistentFiles) { Write-FixtureFile -Path $path }

    $cacheTargets = @(Get-DiligentCacheTargets -RepoRoot $repoRoot -LocalAppData $localAppData -DataRoots @($dataRoot))
    $cachePlan = New-DiligentCleanupPlan -Targets $cacheTargets -AllowedRoots @($repoRoot, $appRoot, $dataRoot)
    $cacheDryRun = Invoke-DiligentCleanupPlan -Plan $cachePlan -DryRun
    Assert-True ($cacheDryRun.Planned.Count -gt 0) 'Cache dry run produced no planned paths.'
    foreach ($path in $disposableFiles) { Assert-True (Test-Path -LiteralPath $path) "Dry run changed fixture path: $path" }
    foreach ($path in $persistentFiles) { Assert-True (Test-Path -LiteralPath $path) "Dry run changed persistent path: $path" }

    $cacheResult = Invoke-DiligentCleanupPlan -Plan $cachePlan
    Assert-True ($cacheResult.Failed.Count -eq 0) 'Cache fixture cleanup reported a failure.'
    foreach ($path in $disposableFiles) {
        if ($path -match '\\cache\\logs\\') { Assert-True (Test-Path -LiteralPath $path) "Cache cleanup unexpectedly removed a log: $path" }
        else { Assert-True (-not (Test-Path -LiteralPath $path)) "Expected cache or generated output remains: $path" }
    }
    foreach ($path in $persistentFiles) { Assert-True (Test-Path -LiteralPath $path) "Cache cleanup removed persistent data: $path" }
    Assert-True (Test-Path -LiteralPath (Join-Path $repoRoot 'runtimes\cache\.gitkeep')) 'Cache cleanup removed the sentinel.'

    $repeat = Invoke-DiligentCleanupPlan -Plan (New-DiligentCleanupPlan -Targets $cacheTargets -AllowedRoots @($repoRoot, $appRoot, $dataRoot))
    Assert-True ($repeat.Failed.Count -eq 0) 'Repeated cleanup failed on missing paths.'

    $uninstallTargets = @(Get-DiligentUninstallTargets -RepoRoot $repoRoot -LocalAppData $localAppData -DataRoots @($dataRoot))
    $uninstallPlan = New-DiligentCleanupPlan -Targets $uninstallTargets -AllowedRoots @($repoRoot, $appRoot, $dataRoot)
    $uninstallDryRun = Invoke-DiligentCleanupPlan -Plan $uninstallPlan -DryRun
    Assert-True ($uninstallDryRun.Planned.Count -gt 0) 'Uninstall dry run produced no planned paths.'
    foreach ($path in $persistentFiles) { Assert-True (Test-Path -LiteralPath $path) "Uninstall dry run changed persistent path: $path" }
    $uninstallResult = Invoke-DiligentCleanupPlan -Plan $uninstallPlan
    Assert-True ($uninstallResult.Failed.Count -eq 0) 'Uninstall fixture cleanup reported a failure.'
    foreach ($path in $persistentFiles) { Assert-True (Test-Path -LiteralPath $path) "Standard uninstall removed persistent data: $path" }
    foreach ($path in @(
        (Join-Path $repoRoot 'app\client\node_modules'),
        (Join-Path $repoRoot 'app\desktop\node_modules'),
        (Join-Path $repoRoot 'app\client\dist'),
        (Join-Path $appRoot 'runtime')
    )) { Assert-True (-not (Test-Path -LiteralPath $path)) "Standard uninstall left generated path: $path" }

    $outsideMarker = Join-Path $outsideRoot 'keep.txt'
    Write-FixtureFile -Path $outsideMarker
    $traversal = New-DiligentCleanupPlan -Targets @([pscustomobject]@{ Path = (Join-Path $repoRoot '..\outside'); Label = 'Traversal fixture' }) -AllowedRoots @($repoRoot)
    Assert-True (@($traversal.InitialResults | Where-Object Status -eq 'Failed').Count -eq 1) 'Traversal target was not rejected.'
    Assert-True (Test-Path -LiteralPath $outsideMarker) 'Traversal validation changed an outside file.'

    $msiEntries = @(
        [pscustomobject]@{ DisplayName = 'DILIGENT Clinical Copilot'; Publisher = 'CTCycle'; UninstallString = 'MsiExec.exe /I{11111111-1111-1111-1111-111111111111}'; InstallLocation = 'C:\Program Files\DILIGENT' },
        [pscustomobject]@{ DisplayName = 'Other Product'; Publisher = 'CTCycle'; UninstallString = 'MsiExec.exe /X{22222222-2222-2222-2222-222222222222}'; InstallLocation = '' },
        [pscustomobject]@{ DisplayName = 'DILIGENT Clinical Copilot'; Publisher = 'Unrelated'; UninstallString = 'MsiExec.exe /X{33333333-3333-3333-3333-333333333333}'; InstallLocation = '' }
    )
    $products = @(Select-DiligentMsiProducts -RegistryEntries $msiEntries)
    Assert-True ($products.Count -eq 1) 'MSI discovery accepted an unrelated product or missed DILIGENT.'
    $startCalls = [Collections.Generic.List[object]]::new()
    $mockStart = { param($filePath, $arguments) [void]$startCalls.Add([pscustomobject]@{ FilePath = $filePath; Arguments = @($arguments) }); [pscustomobject]@{ ExitCode = 0 } }.GetNewClosure()
    $msiResult = Invoke-DiligentMsiUninstall -Product $products[0] -StartProcess $mockStart
    Assert-True ($msiResult.Success -and $startCalls.Count -eq 1) 'Mock MSI uninstall did not complete.'
    Assert-True ($startCalls[0].FilePath -eq 'msiexec.exe' -and $startCalls[0].Arguments -contains '/x' -and $startCalls[0].Arguments -contains $products[0].ProductCode) 'Mock MSI uninstall did not use the allowlisted product code.'
    $mockFail = { param($filePath, $arguments) [pscustomobject]@{ ExitCode = 1603 } }
    Assert-True (-not (Invoke-DiligentMsiUninstall -Product $products[0] -StartProcess $mockFail).Success) 'MSI failure was not reported.'

    $lockedPath = Join-Path $repoRoot 'locked\held.bin'
    Write-FixtureFile -Path $lockedPath
    $lockedStream = [IO.File]::Open($lockedPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $lockedPlan = New-DiligentCleanupPlan -Targets @([pscustomobject]@{ Path = (Split-Path -Parent $lockedPath); Label = 'Locked fixture' }) -AllowedRoots @($repoRoot)
    $lockedResult = Invoke-DiligentCleanupPlan -Plan $lockedPlan
    Assert-True ($lockedResult.Failed.Count -gt 0 -and (Test-Path -LiteralPath $lockedPath)) 'Locked file failure was not reported safely.'
    $lockedStream.Dispose()
    $lockedStream = $null
    $retry = Invoke-DiligentCleanupPlan -Plan (New-DiligentCleanupPlan -Targets @([pscustomobject]@{ Path = (Split-Path -Parent $lockedPath); Label = 'Unlocked fixture' }) -AllowedRoots @($repoRoot))
    Assert-True ($retry.Failed.Count -eq 0 -and -not (Test-Path -LiteralPath $lockedPath)) 'Cleanup did not recover after unlocking the fixture.'

    $deniedFile = Join-Path $aclTarget 'keep.bin'
    Write-FixtureFile -Path $deniedFile
    $aclDenialApplied = $false
    $aclSnapshot = $null
    $icaclsCommand = Get-Command icacls.exe -ErrorAction SilentlyContinue
    $aclTrustee = $null
    try {
        try {
            $aclSnapshot = Get-Acl -LiteralPath $aclTarget -ErrorAction Stop
            $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
            $inheritance = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
            $deny = [Security.AccessControl.FileSystemAccessRule]::new($identity, [Security.AccessControl.FileSystemRights]::ReadAndExecute, $inheritance, [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Deny)
            $deniedAcl = Get-Acl -LiteralPath $aclTarget -ErrorAction Stop
            $deniedAcl.AddAccessRule($deny)
            Set-Acl -LiteralPath $aclTarget -AclObject $deniedAcl -ErrorAction Stop
            $aclDenialApplied = $true
        } catch {
            if (-not $icaclsCommand) { throw }
            $aclTrustee = '*' + [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
            & $icaclsCommand.Source $aclTarget '/deny' ($aclTrustee + ':(OI)(CI)(F)') 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "icacls could not apply the permission-denial fixture (exit $LASTEXITCODE)." }
            $aclDenialApplied = $true
        }
        $aclPlan = New-DiligentCleanupPlan -Targets @([pscustomobject]@{ Path = $aclTarget; Label = 'Permission fixture' }) -AllowedRoots @($fixtureRoot)
        Assert-True (@($aclPlan.InitialResults | Where-Object Status -eq 'Failed').Count -gt 0) 'Permission-denied enumeration was not reported.'
    } catch {
        Write-Host "[INFO] ACL-denial setup unavailable in this security context: $($_.Exception.Message)" -ForegroundColor Yellow
        if ($aclDenialApplied) { throw }
    } finally {
        if ($null -ne $aclTrustee -and $icaclsCommand) {
            & $icaclsCommand.Source $aclTarget '/remove:d' $aclTrustee 2>$null | Out-Null
        } elseif ($null -ne $aclSnapshot) {
            try { Set-Acl -LiteralPath $aclTarget -AclObject $aclSnapshot -ErrorAction SilentlyContinue } catch {}
        }
    }

    $junctionTarget = Join-Path $outsideRoot 'junction-target'
    New-Item -ItemType Directory -Path $junctionTarget -Force | Out-Null
    Write-FixtureFile -Path (Join-Path $junctionTarget 'protected.txt')
    try {
        New-Item -ItemType Junction -Path $junctionPath -Target $junctionTarget -ErrorAction Stop | Out-Null
        $junctionPlan = New-DiligentCleanupPlan -Targets @([pscustomobject]@{ Path = $junctionPath; Label = 'Junction fixture' }) -AllowedRoots @($repoRoot)
        $junctionResult = Invoke-DiligentCleanupPlan -Plan $junctionPlan
        Assert-True ($junctionResult.Skipped.Count -gt 0) 'Junction was not reported as skipped.'
        Assert-True (Test-Path -LiteralPath (Join-Path $junctionTarget 'protected.txt')) 'Cleanup followed a junction outside the allowlist.'
    } catch { Write-Host "[INFO] Junction test unavailable: $($_.Exception.Message)" -ForegroundColor Yellow }
    finally { if (Test-Path -LiteralPath $junctionPath) { try { [IO.Directory]::Delete($junctionPath, $false) } catch {} } }

    Write-Host '[OK] Launcher cleanup fixture tests passed.' -ForegroundColor Green
} finally {
    if ($null -ne $lockedStream) { $lockedStream.Dispose() }
    if ($null -ne $aclSnapshot -and (Test-Path -LiteralPath $aclTarget)) { try { Set-Acl -LiteralPath $aclTarget -AclObject $aclSnapshot -ErrorAction SilentlyContinue } catch {} }
    if (Test-Path -LiteralPath $fixtureRoot) {
        try { Remove-Item -LiteralPath $fixtureRoot -Recurse -Force -ErrorAction Stop }
        catch { Write-Host "[WARN] Could not remove test fixture '$fixtureRoot': $($_.Exception.Message)" -ForegroundColor Yellow }
    }
}

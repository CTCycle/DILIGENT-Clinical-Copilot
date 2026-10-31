Set-StrictMode -Version 2.0

function Get-DiligentFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'Cleanup path cannot be empty.' }
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (Test-Path -LiteralPath $fullPath) {
        try { return (Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop).FullName.TrimEnd('\', '/') }
        catch { }
    }
    return $fullPath.TrimEnd('\', '/')
}

function ConvertTo-DiligentCleanupPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = Get-DiligentFullPath -Path $Path
    $root = [IO.Path]::GetPathRoot($fullPath)
    if (-not $root -or $fullPath.TrimEnd('\', '/') -eq $root.TrimEnd('\', '/')) {
        throw "Refusing to target a filesystem root: $fullPath"
    }
    return $fullPath.TrimEnd('\', '/')
}

function Test-DiligentPathUnderRoot {
    param([string]$Path, [string]$Root)
    $normalizedRoot = $Root.TrimEnd('\', '/')
    $normalizedPath = $Path.TrimEnd('\', '/')
    return $normalizedPath.Equals($normalizedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $normalizedPath.StartsWith($normalizedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $normalizedPath.StartsWith($normalizedRoot + [IO.Path]::AltDirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Test-DiligentReparseChain {
    param([string]$Path)
    $cursor = $Path
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        if (Test-Path -LiteralPath $cursor) {
            try {
                $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $false }
            } catch { return $false }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
    return $true
}

function New-DiligentCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][object[]]$Targets,
        [Parameter(Mandatory = $true)][string[]]$AllowedRoots
    )

    $roots = @($AllowedRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object {
        Get-DiligentFullPath -Path $_
    } | Select-Object -Unique)
    if ($roots.Count -eq 0) { throw 'At least one validated cleanup root is required.' }

    $items = [Collections.Generic.List[object]]::new()
    foreach ($spec in $Targets) {
        $specPropertyNames = if ($spec -is [string]) { @() } else { @($spec.PSObject.Properties.Name) }
        $pathValue = if ($spec -is [string]) { [string]$spec } else { [string]$spec.Path }
        $label = if ($specPropertyNames -contains 'Label' -and $spec.Label) { [string]$spec.Label } else { 'DILIGENT cleanup' }
        $keepRoot = if ($specPropertyNames -contains 'KeepRoot') { [bool]$spec.KeepRoot } else { $false }
        $preserveNames = if ($specPropertyNames -contains 'PreserveNames' -and $spec.PreserveNames) { @($spec.PreserveNames | ForEach-Object { [string]$_ }) } else { @() }
        $preservePaths = if ($specPropertyNames -contains 'PreservePaths' -and $spec.PreservePaths) { @($spec.PreservePaths | ForEach-Object { Get-DiligentFullPath -Path ([string]$_) }) } else { @() }
        $planError = if ($specPropertyNames -contains 'PlanError') { [string]$spec.PlanError } else { '' }
        try { $target = ConvertTo-DiligentCleanupPath -Path $pathValue }
        catch {
            [void]$items.Add([pscustomobject]@{ Path = $pathValue; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = $_.Exception.Message })
            continue
        }

        $allowedRoot = $null
        foreach ($root in $roots) {
            if ((Test-DiligentPathUnderRoot -Path $target -Root $root) -and
                -not $target.Equals($root, [StringComparison]::OrdinalIgnoreCase)) { $allowedRoot = $root; break }
        }
        if (-not $allowedRoot) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = 'Target is outside every allowlisted application root.' })
            continue
        }
        if (-not (Test-DiligentReparseChain -Path $target)) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Skipped'; Message = 'A reparse point exists in the target path; it was not traversed.' })
            continue
        }
        if ($planError) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = $planError })
            continue
        }
        try { $rootItem = Get-Item -LiteralPath $target -Force -ErrorAction Stop }
        catch {
            if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
                [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Skipped'; Message = 'Target does not exist.' })
            } else {
                [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = $_.Exception.Message })
            }
            continue
        }

        $stack = [Collections.Generic.Stack[object]]::new()
        $stack.Push([pscustomobject]@{ Item = $rootItem; IsRoot = $true })
        while ($stack.Count -gt 0) {
            $frame = $stack.Pop()
            $entry = $frame.Item
            $isRoot = [bool]$frame.IsRoot
            $entryPath = [IO.Path]::GetFullPath($entry.FullName).TrimEnd('\', '/')
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Entry'; Status = 'Skipped'; Message = 'Reparse point left untouched.' })
                continue
            }
            $preservedByPath = $false
            foreach ($preservePath in $preservePaths) {
                if (Test-DiligentPathUnderRoot -Path $entryPath -Root $preservePath) { $preservedByPath = $true; break }
            }
            if ($preserveNames -contains [string]$entry.Name -or $preservedByPath) {
                [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Entry'; Status = 'Preserved'; Message = 'Preserve rule.' })
                continue
            }
            if ($entry.PSIsContainer) {
                try {
                    $children = @(Get-ChildItem -LiteralPath $entryPath -Force -ErrorAction Stop)
                    foreach ($child in $children) { $stack.Push([pscustomobject]@{ Item = $child; IsRoot = $false }) }
                } catch {
                    [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Directory'; Status = 'Failed'; Message = "Cannot enumerate directory: $($_.Exception.Message)" })
                }
                if (-not ($isRoot -and $keepRoot)) {
                    [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Directory'; Status = 'Planned'; Message = '' })
                }
            } else {
                [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'File'; Status = 'Planned'; Message = '' })
            }
        }
    }

    $protectedPaths = @($items | Where-Object { $_.Status -in @('Preserved', 'Skipped', 'Failed') } | Select-Object -ExpandProperty Path)
    $planned = @($items | Where-Object {
        if ($_.Status -ne 'Planned') { return $false }
        if ($_.Kind -ne 'Directory') { return $true }
        foreach ($protectedPath in $protectedPaths) {
            if (Test-DiligentPathUnderRoot -Path $protectedPath -Root $_.Path) { return $false }
        }
        return $true
    } | Sort-Object @{ Expression = { $_.Path.Length }; Descending = $true }, @{ Expression = { $_.Path.ToUpperInvariant() }; Descending = $false })
    $initial = @($items | Where-Object Status -ne 'Planned')
    return [pscustomobject]@{ Items = $planned; InitialResults = $initial; AllowedRoots = $roots }
}

function Invoke-DiligentCleanupPlan {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][pscustomobject]$Plan, [switch]$DryRun)

    $results = [Collections.Generic.List[object]]::new()
    foreach ($item in $Plan.InitialResults) { [void]$results.Add($item) }
    foreach ($item in $Plan.Items) {
        if ($DryRun) {
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Planned'; Message = '' })
            continue
        }
        $root = $null
        foreach ($candidate in $Plan.AllowedRoots) {
            if (Test-DiligentPathUnderRoot -Path $item.Path -Root $candidate) { $root = $candidate; break }
        }
        if (-not $root -or -not (Test-DiligentReparseChain -Path $item.Path)) {
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Skipped'; Message = 'Path changed or became a reparse point after planning.' })
            continue
        }
        try {
            $current = Get-Item -LiteralPath $item.Path -Force -ErrorAction Stop
            if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Entry became a reparse point after planning.' }
            if ($current.PSIsContainer -and @(Get-ChildItem -LiteralPath $item.Path -Force -ErrorAction Stop).Count -gt 0) {
                throw 'Directory changed after planning; new contents were left untouched.'
            }
            Remove-Item -LiteralPath $item.Path -Force -ErrorAction Stop
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Cleaned'; Message = '' })
        } catch {
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Failed'; Message = $_.Exception.Message })
        }
    }
    return [pscustomobject]@{
        DryRun = [bool]$DryRun
        Results = @($results)
        Planned = @($results | Where-Object Status -eq 'Planned')
        Cleaned = @($results | Where-Object Status -eq 'Cleaned')
        Skipped = @($results | Where-Object Status -eq 'Skipped')
        Failed = @($results | Where-Object Status -eq 'Failed')
        Preserved = @($results | Where-Object Status -eq 'Preserved')
    }
}

function Get-DiligentCacheTargets {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$LocalAppData,
        [string[]]$DataRoots = @(),
        [switch]$IncludeLogs
    )

    $targets = [Collections.Generic.List[object]]::new()
    $add = { param([string]$Path, [string]$Label, [switch]$KeepRoot, [string[]]$PreservePaths = @(), [string]$PlanError)
        if (-not [string]::IsNullOrWhiteSpace($Path)) {
            [void]$targets.Add([pscustomobject]@{ Path = $Path; Label = $Label; KeepRoot = [bool]$KeepRoot; PreserveNames = @('.gitkeep'); PreservePaths = @($PreservePaths); PlanError = $PlanError })
        }
    }

    $runtimeCache = Join-Path $RepoRoot 'runtimes\cache'
    $runtimePreserve = if ($IncludeLogs) { @() } else { @((Join-Path $runtimeCache 'logs')) }
    & $add $runtimeCache 'Repository runtime and tool caches' -KeepRoot -PreservePaths $runtimePreserve
    foreach ($path in @(
        (Join-Path $RepoRoot 'app\server\runtimes\cache'),
        (Join-Path $RepoRoot 'app\client\.angular\cache'),
        (Join-Path $RepoRoot 'app\client\node_modules\.cache'),
        (Join-Path $RepoRoot 'app\client\node_modules\.vite'),
        (Join-Path $RepoRoot 'app\client\.cache'),
        (Join-Path $RepoRoot 'app\client\dist'),
        (Join-Path $RepoRoot 'app\client\dist-ssr'),
        (Join-Path $RepoRoot 'app\client\out-tsc'),
        (Join-Path $RepoRoot 'app\client\coverage'),
        (Join-Path $RepoRoot 'app\desktop\node_modules\.cache'),
        (Join-Path $RepoRoot 'app\desktop\node_modules\.vite'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\target'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\gen'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\generated'),
        (Join-Path $RepoRoot 'assets\QA\desktop-release-staging'),
        (Join-Path $RepoRoot 'htmlcov'),
        (Join-Path $RepoRoot '.coverage')
    )) { & $add $path 'Generated cache or build output' }

    foreach ($coverage in Get-ChildItem -LiteralPath $RepoRoot -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\.coverage\.' }) {
        & $add $coverage.FullName 'Parallel Python coverage output'
    }

    $cacheNames = @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.cache', '.vite', '.angular-cache', 'coverage', 'htmlcov', 'out-tsc')
    $pruneNames = @('.git', '.venv', 'venv', 'node_modules', 'dist', 'dist-ssr', 'target', 'runtimes', 'release', 'assets', 'fixtures')
    $scanRoots = @(
        (Join-Path $RepoRoot 'app\server'),
        (Join-Path $RepoRoot 'app\client'),
        (Join-Path $RepoRoot 'app\desktop\build'),
        (Join-Path $RepoRoot 'app\tests')
    )
    foreach ($scanRoot in $scanRoots) {
        if (-not (Test-Path -LiteralPath $scanRoot -PathType Container)) { continue }
        if (-not (Test-DiligentReparseChain -Path $scanRoot)) { & $add $scanRoot 'Nested cache scan skipped'; continue }
        $stack = [Collections.Generic.Stack[string]]::new()
        $stack.Push([IO.Path]::GetFullPath($scanRoot))
        while ($stack.Count -gt 0) {
            $directory = $stack.Pop()
            try { $entries = @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) }
            catch {
                & $add $directory 'Nested cache scan failed' -KeepRoot -PlanError $_.Exception.Message
                continue
            }
            foreach ($entry in $entries) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    if (($entry.PSIsContainer -and $cacheNames -contains $entry.Name) -or $entry.Extension -in @('.pyc', '.pyo')) {
                        & $add $entry.FullName 'Cache reparse point left untouched'
                    }
                    continue
                }
                if ($entry.PSIsContainer) {
                    if ($cacheNames -contains $entry.Name) { & $add $entry.FullName 'Nested Python or frontend cache'; continue }
                    if ($pruneNames -contains $entry.Name) { continue }
                    $stack.Push($entry.FullName)
                } elseif ($entry.Extension -in @('.pyc', '.pyo') -or $entry.Name -eq '.coverage' -or $entry.Name -match '^\.coverage\.') {
                    & $add $entry.FullName 'Python bytecode or coverage output'
                }
            }
        }
    }

    $legacyQaRoot = Join-Path $RepoRoot 'app\tests\assets\QA'
    if (Test-Path -LiteralPath $legacyQaRoot -PathType Container) {
        $stack = [Collections.Generic.Stack[string]]::new()
        $stack.Push([IO.Path]::GetFullPath($legacyQaRoot))
        while ($stack.Count -gt 0) {
            $directory = $stack.Pop()
            try { $entries = @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) }
            catch { & $add $directory 'Legacy QA cache scan failed' -KeepRoot -PlanError $_.Exception.Message; continue }
            foreach ($entry in $entries) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    if ($entry.PSIsContainer -and $entry.Name -match '^(?:\.pytest_cache|pytest-cache|pytest-root-cache|pytest-cache-[A-Za-z0-9._-]+)$') {
                        & $add $entry.FullName 'Legacy QA cache reparse point left untouched'
                    }
                    continue
                }
                if ($entry.PSIsContainer -and $entry.Name -match '^(?:\.pytest_cache|pytest-cache|pytest-root-cache|pytest-cache-[A-Za-z0-9._-]+)$') {
                    & $add $entry.FullName 'Ignored disposable pytest cache'
                } elseif ($entry.PSIsContainer) { $stack.Push($entry.FullName) }
            }
        }
    }

    foreach ($bytecodeRoot in @(
        (Join-Path $RepoRoot 'app\server\.venv'),
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $RepoRoot 'runtimes\python')
    )) {
        if (-not (Test-Path -LiteralPath $bytecodeRoot -PathType Container)) { continue }
        if (-not (Test-DiligentReparseChain -Path $bytecodeRoot)) { & $add $bytecodeRoot 'Python bytecode scan skipped'; continue }
        $stack = [Collections.Generic.Stack[string]]::new()
        $stack.Push([IO.Path]::GetFullPath($bytecodeRoot))
        while ($stack.Count -gt 0) {
            $directory = $stack.Pop()
            try { $entries = @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) }
            catch { & $add $directory 'Python environment cache scan failed' -KeepRoot -PlanError $_.Exception.Message; continue }
            foreach ($entry in $entries) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    if (($entry.PSIsContainer -and $entry.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.cache')) -or $entry.Extension -in @('.pyc', '.pyo')) {
                        & $add $entry.FullName 'Python environment cache reparse point left untouched'
                    }
                    continue
                }
                if ($entry.PSIsContainer) {
                    if ($entry.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.cache')) {
                        & $add $entry.FullName 'Python environment cache'
                    } else { $stack.Push($entry.FullName) }
                } elseif ($entry.Extension -in @('.pyc', '.pyo')) { & $add $entry.FullName 'Python environment bytecode' }
            }
        }
    }

    foreach ($dataRoot in @($DataRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        foreach ($fileSpec in @(
            @{ Directory = $dataRoot; Pattern = '^\.database_config\.json\..+\.tmp$'; Label = 'Stale database configuration temporary' },
            @{ Directory = (Join-Path $dataRoot 'settings'); Pattern = '^\.\.env\.\d+\.tmp$'; Label = 'Stale packaged settings temporary' }
        )) {
            if (-not (Test-Path -LiteralPath $fileSpec.Directory -PathType Container)) { continue }
            try {
                foreach ($file in Get-ChildItem -LiteralPath $fileSpec.Directory -File -Force -ErrorAction Stop | Where-Object { $_.Name -match $fileSpec.Pattern }) {
                    & $add $file.FullName $fileSpec.Label
                }
            } catch { & $add $fileSpec.Directory "$($fileSpec.Label) scan failed" -KeepRoot -PlanError $_.Exception.Message }
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($LocalAppData)) {
        $packagedCache = Join-Path (Join-Path (Join-Path $LocalAppData 'DILIGENT') 'data') 'cache'
        $preserveLogs = if ($IncludeLogs) { @() } else { @((Join-Path $packagedCache 'logs')) }
        & $add $packagedCache 'Packaged runtime and embedding caches' -KeepRoot -PreservePaths $preserveLogs
        $runtimeRoot = Join-Path (Join-Path $LocalAppData 'DILIGENT') 'runtime'
        if (Test-Path -LiteralPath $runtimeRoot -PathType Container) {
            try {
                foreach ($versionRoot in Get-ChildItem -LiteralPath $runtimeRoot -Directory -Force -ErrorAction Stop) {
                    if (($versionRoot.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                        & $add $versionRoot.FullName 'Packaged runtime staging scan skipped'
                        continue
                    }
                    foreach ($temporary in Get-ChildItem -LiteralPath $versionRoot.FullName -Directory -Force -ErrorAction Stop | Where-Object { $_.Name -match '^\.extract-[0-9a-fA-F-]+$' }) {
                        & $add $temporary.FullName 'Incomplete packaged runtime extraction'
                    }
                }
            } catch { & $add $runtimeRoot 'Packaged runtime temporary scan failed' -KeepRoot -PlanError $_.Exception.Message }
        }
    }

    $unique = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($target in $targets) {
        $key = [IO.Path]::GetFullPath([string]$target.Path)
        if (-not $unique.ContainsKey($key)) { $unique.Add($key, $target) }
    }
    return @($unique.Values)
}

function Get-DiligentUninstallTargets {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$RepoRoot, [string]$LocalAppData, [string[]]$DataRoots = @())
    $targets = [Collections.Generic.List[object]]::new()
    [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $RepoRoot 'runtimes'); Label = 'Launcher-managed portable runtimes and caches'; KeepRoot = $false; PreserveNames = @('.gitkeep') })
    foreach ($path in @(
        (Join-Path $RepoRoot 'app\server\.venv'),
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $RepoRoot 'app\client\node_modules'),
        (Join-Path $RepoRoot 'app\client\dist'),
        (Join-Path $RepoRoot 'app\desktop\node_modules'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\target'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\gen'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\generated'),
        (Join-Path $RepoRoot 'assets\QA\desktop-release-staging')
    )) { [void]$targets.Add([pscustomobject]@{ Path = $path; Label = 'Repository dependency or generated build output'; PreserveNames = @('.gitkeep') }) }

    foreach ($cacheTarget in @(Get-DiligentCacheTargets -RepoRoot $RepoRoot -LocalAppData $LocalAppData -DataRoots $DataRoots -IncludeLogs)) {
        [void]$targets.Add($cacheTarget)
    }
    if (-not [string]::IsNullOrWhiteSpace($LocalAppData)) {
        $diligentRoot = Join-Path $LocalAppData 'DILIGENT'
        [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $diligentRoot 'runtime'); Label = 'Packaged extracted application runtime' })
        [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $diligentRoot 'data\state\desktop-backend-ready.json'); Label = 'Packaged transient runtime state' })
    }
    $unique = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($target in $targets) {
        $key = [IO.Path]::GetFullPath([string]$target.Path)
        if (-not $unique.ContainsKey($key)) { $unique.Add($key, $target) }
    }
    $filtered = [Collections.Generic.List[object]]::new()
    foreach ($target in @($unique.Values | Sort-Object { [IO.Path]::GetFullPath([string]$_.Path).Length })) {
        $path = [IO.Path]::GetFullPath([string]$target.Path)
        $covered = $false
        foreach ($existing in $filtered) {
            $parent = [IO.Path]::GetFullPath([string]$existing.Path)
            if ((Test-DiligentPathUnderRoot -Path $path -Root $parent) -and
                -not $path.Equals($parent, [StringComparison]::OrdinalIgnoreCase)) { $covered = $true; break }
        }
        if (-not $covered) { [void]$filtered.Add($target) }
    }
    return @($filtered)
}

function Select-DiligentMsiProducts {
    [CmdletBinding()]
    param([object[]]$RegistryEntries = @())
    $products = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $RegistryEntries) {
        if ($null -eq $entry) { continue }
        $properties = $entry.PSObject.Properties
        $displayName = if ($properties['DisplayName']) { [string]$properties['DisplayName'].Value } else { '' }
        $publisher = if ($properties['Publisher']) { [string]$properties['Publisher'].Value } else { '' }
        if ($displayName -ne 'DILIGENT Clinical Copilot' -or $publisher -ne 'CTCycle') { continue }
        $uninstall = if ($properties['UninstallString']) { [string]$properties['UninstallString'].Value } else { '' }
        $childName = if ($properties['PSChildName']) { [string]$properties['PSChildName'].Value } else { '' }
        $guidPattern = '\{[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\}'
        $match = [regex]::Match($uninstall, $guidPattern)
        $productCode = if ($match.Success) { $match.Value } elseif ($childName -match "^$guidPattern$") { $childName } else { '' }
        if (-not $productCode -or $products.ContainsKey($productCode)) { continue }
        $installLocation = if ($properties['InstallLocation']) { [string]$properties['InstallLocation'].Value } else { '' }
        $products.Add($productCode, [pscustomobject]@{ DisplayName = $displayName; Publisher = $publisher; ProductCode = $productCode; InstallLocation = $installLocation })
    }
    return @($products.Values | Sort-Object DisplayName, ProductCode)
}

function Invoke-DiligentMsiUninstall {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][pscustomobject]$Product, [scriptblock]$StartProcess)
    if ($Product.Publisher -ne 'CTCycle' -or $Product.DisplayName -ne 'DILIGENT Clinical Copilot' -or
        $Product.ProductCode -notmatch '^\{[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\}$') {
        return [pscustomobject]@{ Product = $Product; Success = $false; RestartRequired = $false; ExitCode = $null; Message = 'Product did not pass the DILIGENT MSI allowlist.' }
    }
    $arguments = @('/x', [string]$Product.ProductCode, '/passive', '/norestart')
    try {
        $process = if ($StartProcess) { & $StartProcess 'msiexec.exe' $arguments }
        else { Start-Process -FilePath 'msiexec.exe' -ArgumentList $arguments -Wait -PassThru -ErrorAction Stop }
        $exitCode = [int]$process.ExitCode
        return [pscustomobject]@{ Product = $Product; Success = $exitCode -in @(0, 3010); RestartRequired = $exitCode -eq 3010; ExitCode = $exitCode; Message = if ($exitCode -in @(0, 3010)) { '' } else { "msiexec exit $exitCode" } }
    } catch { return [pscustomobject]@{ Product = $Product; Success = $false; RestartRequired = $false; ExitCode = $null; Message = $_.Exception.Message } }
}

Export-ModuleMember -Function New-DiligentCleanupPlan, Invoke-DiligentCleanupPlan, Get-DiligentCacheTargets, Get-DiligentUninstallTargets, Select-DiligentMsiProducts, Invoke-DiligentMsiUninstall

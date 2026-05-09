[CmdletBinding()]
param(
  [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$clientDir = Join-Path $repoRoot "app\client"
$releaseDir = Join-Path $clientDir "src-tauri\target\release"
$bundleDir = Join-Path $releaseDir "bundle"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $outputDir = Join-Path $repoRoot "release\windows"
} else {
  $outputDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}

$installersDir = Join-Path $outputDir "installers"
$portableDir = Join-Path $outputDir "portable"

if (-not (Test-Path $bundleDir)) {
  throw "Bundle directory not found. Run 'npm run tauri:build' first. Missing: $bundleDir"
}

if (Test-Path $outputDir) {
  Remove-Item -Recurse -Force $outputDir
}

New-Item -ItemType Directory -Path $installersDir -Force | Out-Null
New-Item -ItemType Directory -Path $portableDir -Force | Out-Null

$installerArtifacts = @()

$nsisDir = Join-Path $bundleDir "nsis"
if (Test-Path $nsisDir) {
  $nsisFiles = Get-ChildItem -Path $nsisDir -Filter "*.exe" -File
  foreach ($file in $nsisFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$msiDir = Join-Path $bundleDir "msi"
if (Test-Path $msiDir) {
  $msiFiles = Get-ChildItem -Path $msiDir -Filter "*.msi" -File
  foreach ($file in $msiFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$portableExeCandidates = Get-ChildItem -Path $releaseDir -Filter "*.exe" -File |
  Where-Object { $_.Name -notmatch "(?i)(setup|installer|uninstall|updater)" }

if ($portableExeCandidates.Count -eq 0) {
  throw "No portable desktop executable found in release directory: $releaseDir"
}

foreach ($file in $portableExeCandidates) {
  Copy-Item -Path $file.FullName -Destination $portableDir -Force
}

$portablePayloadSourceDir = Join-Path $releaseDir "r"
$portablePayloadDestinationDir = Join-Path $portableDir "r"
if (Test-Path $portablePayloadSourceDir) {
  Copy-Item -Path $portablePayloadSourceDir -Destination $portablePayloadDestinationDir -Recurse -Force
}

$requiredPortablePaths = @(
  (Join-Path $portableDir "r"),
  (Join-Path $portableDir "r\server"),
  (Join-Path $portableDir "r\scripts"),
  (Join-Path $portableDir "r\settings"),
  (Join-Path $portableDir "r\client\dist"),
  (Join-Path $portableDir "r\resources\models"),
  (Join-Path $portableDir "r\resources\sources"),
  (Join-Path $portableDir "r\runtimes\uv\uv.exe"),
  (Join-Path $portableDir "r\runtimes\python\python.exe"),
  (Join-Path $portableDir "r\runtimes\nodejs\node.exe"),
  (Join-Path $portableDir "r\runtimes\nodejs\npm.cmd"),
  (Join-Path $portableDir "r\runtimes\uv.lock"),
  (Join-Path $portableDir "r\pyproject.toml")
)

foreach ($requiredPath in $requiredPortablePaths) {
  if (-not (Test-Path $requiredPath)) {
    throw "Portable export is incomplete. Missing required payload path: $requiredPath"
  }
}

$instructions = @"
DILIGENT desktop build output

1) Preferred for users:
   Open installers\ and run the setup executable (.exe) or .msi.

2) Portable executable:
   portable\ contains the app .exe and the required runtime resource payload.
   Keep the exported contents together in the same directory.

Generated from:
$bundleDir
"@
Set-Content -Path (Join-Path $outputDir "README.txt") -Value $instructions -Encoding ascii

Write-Host "[OK] Exported Windows artifacts to: $outputDir"
Write-Host "[INFO] Installers:"
if ($installerArtifacts.Count -eq 0) {
  Write-Host " - none found"
} else {
  $installerArtifacts | ForEach-Object { Write-Host " - $_" }
}
Write-Host "[INFO] Portable executables:"
$portableFiles = Get-ChildItem -Path $portableDir -Filter "*.exe" -File
if ($portableFiles.Count -eq 0) {
  Write-Host " - none found"
} else {
  $portableFiles | ForEach-Object { Write-Host " - $($_.FullName)" }
}

# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string[]]$Path,
    [ValidateSet('Sign', 'Verify')][string]$Action = 'Verify'
)

$ErrorActionPreference = 'Stop'
$expected = ($env:WINDOWS_CODESIGN_CERT_THUMBPRINT -replace '\s', '').ToUpperInvariant()
if ($expected -notmatch '^[0-9A-F]{40}$') {
    throw 'WINDOWS_CODESIGN_CERT_THUMBPRINT must identify the trusted publisher certificate.'
}
$command = Get-Command signtool.exe -ErrorAction SilentlyContinue
$signtool = if ($command) { $command.Source } else {
    Get-ChildItem "${env:ProgramFiles(x86)}/Windows Kits/10/bin/*/x64/signtool.exe" -File -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $signtool) { throw 'Windows SDK signtool.exe is required.' }
$pfxPath = $null
try {
    if ($Action -eq 'Sign') {
        foreach ($name in @('WINDOWS_CODESIGN_PFX_BASE64', 'WINDOWS_CODESIGN_PFX_PASSWORD', 'WINDOWS_CODESIGN_TIMESTAMP_URL')) {
            if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
                throw "Missing signing configuration: $name"
            }
        }
        if ($env:WINDOWS_CODESIGN_TIMESTAMP_URL -notmatch '^https://') {
            throw 'The signing timestamp URL must use HTTPS.'
        }
        $pfxPath = Join-Path ([IO.Path]::GetTempPath()) ("diligent-signing-$([guid]::NewGuid()).pfx")
        try { $bytes = [Convert]::FromBase64String($env:WINDOWS_CODESIGN_PFX_BASE64) }
        catch { throw 'The signing certificate is not valid base64.' }
        try { [IO.File]::WriteAllBytes($pfxPath, $bytes) }
        finally { [Array]::Clear($bytes, 0, $bytes.Length) }
    }
    foreach ($artifact in $Path) {
        if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) { throw "Missing signing artifact: $artifact" }
        if ($Action -eq 'Sign') {
            & $signtool sign /fd SHA256 /td SHA256 /tr $env:WINDOWS_CODESIGN_TIMESTAMP_URL /f $pfxPath /p $env:WINDOWS_CODESIGN_PFX_PASSWORD $artifact
            if ($LASTEXITCODE -ne 0) { throw "Authenticode signing failed for $artifact." }
        }
        $signature = Get-AuthenticodeSignature -LiteralPath $artifact
        if ($signature.Status -ne 'Valid' -or $null -eq $signature.SignerCertificate -or
            $signature.SignerCertificate.Thumbprint.ToUpperInvariant() -ne $expected -or
            $null -eq $signature.TimeStamperCertificate) {
            throw "Trusted, timestamped publisher signature is required for $artifact (status: $($signature.Status))."
        }
        & $signtool verify /pa /all /tw $artifact
        if ($LASTEXITCODE -ne 0) { throw "Signature verification failed for $artifact." }
        [pscustomobject]@{
            file = Split-Path -Leaf $artifact
            sha256 = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant()
            signer = $expected
            timestamped = $true
        }
    }
}
finally {
    if ($pfxPath -and (Test-Path -LiteralPath $pfxPath)) { Remove-Item -LiteralPath $pfxPath -Force }
}

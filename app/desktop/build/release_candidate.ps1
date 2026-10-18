# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('Source', 'Record', 'Candidate', 'Verify', 'Publish')][string]$Action,
    [Parameter(Mandatory = $true)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [string]$CandidateRunId,
    [string]$ArtifactDirectory = 'release'
)

$ErrorActionPreference = 'Stop'
$repo = $env:GITHUB_REPOSITORY
$sha = $env:GITHUB_SHA
$tag = "v$Version"
$prefix = "DILIGENT-v$Version-windows-x64"
$files = @("$prefix-portable.exe", "$prefix.msi", "$prefix.sha256")
$requiredGates = @('release-preflight', 'release-persistence-preflight', 'release-live-provider-e2e', 'release-security-preflight')

function Read-GitHub([string]$endpoint) {
    $result = & gh api $endpoint
    if ($LASTEXITCODE -ne 0) { throw "GitHub API read failed: $endpoint" }
    return ($result | ConvertFrom-Json)
}

function Assert-SynchronizedSource {
    if ($env:GITHUB_REF -ne 'refs/heads/main' -or $sha -notmatch '^[0-9a-f]{40}$') {
        throw 'Release preparation/publication must be dispatched from main.'
    }
    foreach ($branch in @('main', 'develop')) {
        $current = Read-GitHub "repos/$repo/branches/$branch"
        if ($current.commit.sha -ne $sha) { throw "$branch is not synchronized at $sha." }
    }
}

function Assert-Source {
    Assert-SynchronizedSource
    $tags = @(Read-GitHub "repos/$repo/git/matching-refs/tags/$tag")
    if (@($tags | Where-Object ref -eq "refs/tags/$tag").Count -gt 0) { throw "$tag already exists; refusing duplicate publication." }
    # Listing is deliberate: authentication/network failures must never mean absence.
    $releasesJson = & gh api --paginate "repos/$repo/releases?per_page=100" --jq '.[].tag_name'
    if ($LASTEXITCODE -ne 0) { throw 'Unable to establish release absence.' }
    if (@($releasesJson) -contains $tag) { throw "A release for $tag already exists." }
    $package = Get-Content app/client/package.json -Raw | ConvertFrom-Json
    $desktop = Get-Content app/desktop/package.json -Raw | ConvertFrom-Json
    $tauri = Get-Content app/desktop/src-tauri/tauri.conf.json -Raw | ConvertFrom-Json
    if ($package.version -ne $Version -or $desktop.version -ne $Version -or $tauri.version -ne $Version) {
        throw 'The requested release version does not match the source manifests.'
    }
    foreach ($path in @('app/client/package-lock.json', 'app/desktop/package-lock.json')) {
        $lock = Get-Content $path -Raw | ConvertFrom-Json -AsHashtable
        if ($lock.version -ne $Version -or $lock.packages[''].version -ne $Version) { throw "Release lockfile version mismatch: $path" }
    }
    foreach ($path in @('app/server/pyproject.toml', 'app/desktop/src-tauri/Cargo.toml')) {
        $match = [regex]::Match((Get-Content $path -Raw), '(?m)^version\s*=\s*"([^"]+)"')
        if (-not $match.Success -or $match.Groups[1].Value -ne $Version) { throw "Release manifest version mismatch: $path" }
    }
}

function Assert-Environment {
    $environment = Read-GitHub "repos/$repo/environments/desktop-release"
    $reviewRules = @($environment.protection_rules | Where-Object type -eq 'required_reviewers')
    if ($reviewRules.Count -ne 1 -or @($reviewRules[0].reviewers).Count -eq 0 -or
        $environment.can_admins_bypass -ne $false) {
        throw 'desktop-release must require reviewer approval and disable administrator bypass.'
    }
    if ([string]::IsNullOrWhiteSpace($env:OPERATOR_EVIDENCE)) {
        throw 'Publication requires the operator evidence reference for native and clean-machine validation.'
    }
}

function Assert-CandidateRun {
    if ($CandidateRunId -notmatch '^[1-9][0-9]*$') { throw 'A successful prepare run ID is required.' }
    $run = Read-GitHub "repos/$repo/actions/runs/$CandidateRunId"
    if ($run.path -ne '.github/workflows/release.yml' -or $run.event -ne 'workflow_dispatch' -or
        $run.head_branch -ne 'main' -or $run.head_sha -ne $sha -or
        $run.status -ne 'completed' -or $run.conclusion -ne 'success') {
        throw 'The candidate must come from a successful preparation of this exact main SHA.'
    }
    $jobs = (Read-GitHub "repos/$repo/actions/runs/$CandidateRunId/jobs?filter=latest&per_page=100").jobs
    foreach ($gate in @($requiredGates) + @('prepare')) {
        $job = @($jobs | Where-Object name -eq $gate)
        if ($job.Count -ne 1 -or $job[0].conclusion -ne 'success') { throw "Candidate gate $gate did not pass." }
    }
    return $run
}

function Assert-Artifacts($receipt, [string]$directory) {
    foreach ($file in $files) {
        $path = Join-Path $directory $file
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing candidate file: $file" }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($receipt.files.$file -ne $hash) { throw "Candidate digest mismatch: $file" }
    }
    $manifest = Get-Content (Join-Path $directory "$prefix.sha256")
    foreach ($file in $files[0..1]) {
        $records = @($manifest | Where-Object { $_ -match ('^' + $receipt.files.$file + '\s+\*?' + [regex]::Escape($file) + '$') })
        if ($records.Count -ne 1) { throw "Checksum manifest must contain exactly one matching entry for $file." }
        & "$PSScriptRoot/sign_artifact.ps1" -Path (Join-Path $directory $file) | Out-Null
    }
}

switch ($Action) {
    'Source' { Assert-Source }
    'Candidate' {
        Assert-Source
        Assert-Environment
        $run = Assert-CandidateRun
        "attempt=$($run.run_attempt)" >> $env:GITHUB_OUTPUT
        "candidate_url=$($run.html_url)" >> $env:GITHUB_OUTPUT
        "Candidate: $($run.html_url)`nSource: $sha`nOperator evidence: $env:OPERATOR_EVIDENCE`nApproval attests to populated native workflows and clean Windows 11/WebView2 checks against the candidate hashes." >> $env:GITHUB_STEP_SUMMARY
    }
    'Record' {
        $gates = $env:RELEASE_GATE_RESULTS | ConvertFrom-Json
        $gateResults = @{}
        foreach ($gate in $requiredGates) {
            if ($gates.$gate.result -ne 'success') { throw "Required preparation gate $gate did not pass." }
            $gateResults[$gate] = 'success'
        }
        $smokePath = Join-Path $ArtifactDirectory 'qa/smoke.json'
        $smoke = Get-Content $smokePath -Raw | ConvertFrom-Json
        if ($smoke.success -ne $true -or $smoke.version -ne $Version -or
            $smoke.msi_reinstall.product_absent -ne $true -or
            $smoke.msi_upgrade.sessions_versions_settings_preserved -ne $true -or
            $null -eq $smoke.portable_deep_path -or $smoke.deep_path.executable_length -ne 230 -or
            $smoke.deep_path.localappdata_length -ne 107) { throw 'The full Windows lifecycle/deep-path receipt is required.' }
        $hashes = @{}
        foreach ($file in $files) { $hashes[$file] = (Get-FileHash (Join-Path $ArtifactDirectory $file) -Algorithm SHA256).Hash.ToLowerInvariant() }
        $receipt = [pscustomobject]@{
            version = $Version; source_sha = $sha; run_id = $env:GITHUB_RUN_ID
            run_attempt = $env:GITHUB_RUN_ATTEMPT; gates = $gateResults; files = $hashes
            smoke_sha256 = (Get-FileHash $smokePath -Algorithm SHA256).Hash.ToLowerInvariant()
            signer = $env:WINDOWS_CODESIGN_CERT_THUMBPRINT
        }
        Assert-Artifacts $receipt $ArtifactDirectory
        $receipt | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $ArtifactDirectory 'candidate.json') -Encoding utf8
        "Signed candidate $sha, run $env:GITHUB_RUN_ID`n$($receipt.files | ConvertTo-Json)" >> $env:GITHUB_STEP_SUMMARY
    }
    { $_ -in 'Verify', 'Publish' } {
        Assert-Source
        Assert-Environment
        $run = Assert-CandidateRun
        $receipt = Get-Content (Join-Path $ArtifactDirectory 'candidate.json') -Raw | ConvertFrom-Json
        if ($receipt.version -ne $Version -or $receipt.source_sha -ne $sha -or
            [string]$receipt.run_id -ne $CandidateRunId -or [int]$receipt.run_attempt -ne $run.run_attempt -or
            $receipt.signer -ne $env:WINDOWS_CODESIGN_CERT_THUMBPRINT) { throw 'Candidate receipt provenance does not match the selected run.' }
        foreach ($gate in $requiredGates) {
            if ($receipt.gates.$gate -ne 'success') { throw "Candidate receipt lacks $gate." }
        }
        $smokePath = Join-Path $ArtifactDirectory 'qa/smoke.json'
        if ((Get-FileHash $smokePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.smoke_sha256) { throw 'Windows lifecycle receipt digest mismatch.' }
        Assert-Artifacts $receipt $ArtifactDirectory
        if ($Action -eq 'Publish') {
            # These checks run again immediately before the first public mutation.
            Assert-Source
            & gh api --method POST "repos/$repo/git/refs" -f "ref=refs/tags/$tag" -f "sha=$sha" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the verified release tag.' }
            $notes = Join-Path $ArtifactDirectory 'release-notes.md'
            @("Validated source: $sha", "Candidate: $($run.html_url)", "Operator evidence: $env:OPERATOR_EVIDENCE", '', 'The attached binaries are the signed, tested candidate; no rebuild was performed.') | Set-Content $notes
            & gh release create $tag --repo $repo --verify-tag --draft --title "DILIGENT Clinical Copilot $tag" --notes-file $notes
            if ($LASTEXITCODE -ne 0) { throw 'Draft release creation failed; the tag remains for investigation.' }
            foreach ($file in @($files) + @('candidate.json')) {
                & gh release upload $tag (Join-Path $ArtifactDirectory $file) --repo $repo
                if ($LASTEXITCODE -ne 0) { throw "Draft asset upload failed: $file" }
            }
            $download = Join-Path 'assets/QA' "published-verification-$env:GITHUB_RUN_ID"
            New-Item -ItemType Directory -Path $download -Force | Out-Null
            & gh release download $tag --repo $repo --dir $download
            if ($LASTEXITCODE -ne 0) { throw 'Draft asset readback failed.' }
            Assert-Artifacts $receipt $download
            if ((Get-FileHash (Join-Path $download 'candidate.json')).Hash -ne (Get-FileHash (Join-Path $ArtifactDirectory 'candidate.json')).Hash) {
                throw 'Published candidate provenance receipt differs from preparation.'
            }
            # Launch the downloaded bytes, not a newly built or original local EXE.
            $releaseRoot = Join-Path (Resolve-Path '.').Path 'release'
            if ((Resolve-Path -LiteralPath $ArtifactDirectory).Path -ne $releaseRoot) { throw 'Publication expects the release artifact directory.' }
            foreach ($file in $files) { Copy-Item -LiteralPath (Join-Path $download $file) -Destination (Join-Path $releaseRoot $file) -Force }
            & "$PSScriptRoot/smoke_release.ps1" -Version $Version -DesktopTarget Portable -RequireSignature -ReportDirectory (Join-Path (Resolve-Path 'assets/QA').Path "published-smoke-$env:GITHUB_RUN_ID") | Out-Null
            Assert-SynchronizedSource
            & gh release edit $tag --repo $repo --draft=false --latest
            if ($LASTEXITCODE -ne 0) { throw 'Publishing the verified draft release failed.' }
            $publicDownload = Join-Path 'assets/QA' "public-readback-$env:GITHUB_RUN_ID"
            New-Item -ItemType Directory -Path $publicDownload -Force | Out-Null
            & gh release download $tag --repo $repo --dir $publicDownload
            if ($LASTEXITCODE -ne 0) { throw 'Public release download failed.' }
            Assert-Artifacts $receipt $publicDownload
            $published = Read-GitHub "repos/$repo/releases/tags/$tag"
            $ref = Read-GitHub "repos/$repo/git/ref/tags/$tag"
            $latest = Read-GitHub "repos/$repo/releases/latest"
            if ($published.draft -or $ref.object.sha -ne $sha -or $latest.tag_name -ne $tag) { throw 'External release/tag/latest verification failed.' }
            "Published: $($published.html_url)" >> $env:GITHUB_STEP_SUMMARY
        }
    }
}

param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$outDirectory = $PSScriptRoot
$planPath = Join-Path $outDirectory 'cleanup_plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$workspaceRoot = [System.IO.Path]::GetFullPath('D:\dataset\experiment').TrimEnd('\')
if ($plan.root -ne $workspaceRoot) { throw 'Unexpected cleanup workspace.' }
$allowedExtensions = @('.pth')

function Get-CheckedFile($record) {
    $absolutePath = [System.IO.Path]::GetFullPath([string]$record.path)
    if (-not $absolutePath.StartsWith($workspaceRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Out-of-workspace path: $absolutePath"
    }
    if ($absolutePath -ne [string]$record.path) { throw 'Plan path is not already absolute and normalized.' }
    $file = Get-Item -LiteralPath $absolutePath -Force
    if ($file.PSIsContainer -or $allowedExtensions -notcontains $file.Extension.ToLowerInvariant()) {
        throw "Not a planned checkpoint file: $absolutePath"
    }
    $node = $file
    while ($null -ne $node -and $node.FullName -ne $workspaceRoot) {
        if (($node.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse point is not allowed: $($node.FullName)"
        }
        if ($node -is [System.IO.FileInfo]) { $node = $node.Directory } else { $node = $node.Parent }
    }
    if ($null -eq $node) { throw 'Workspace ancestry check failed.' }
    $unixNs = ([Int64]$file.LastWriteTimeUtc.Ticks - [Int64]621355968000000000) * [Int64]100
    if ([Int64]$file.Length -ne [Int64]$record.bytes -or $unixNs -ne [Int64]$record.mtime_ns) {
        throw "File changed since inventory: $absolutePath"
    }
    return $file
}

$deletePaths = @($plan.delete | ForEach-Object { [string]$_.path })
if (@($deletePaths | Select-Object -Unique).Count -ne $deletePaths.Count) { throw 'Duplicate deletion path.' }
$critical = @{}
foreach ($property in $plan.critical_expected_sha256.PSObject.Properties) {
    $critical[$property.Name] = [string]$property.Value
}
$audit = [ordered]@{
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    root = $workspaceRoot
    plan_sha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
    delete = @()
    keep = @()
    deletion_started = $false
    complete = $false
}
foreach ($record in @($plan.keep) + @($plan.delete)) {
    $file = Get-CheckedFile $record
    $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($critical.ContainsKey($file.FullName) -and $critical[$file.FullName] -ne $digest) {
        throw "Protected checkpoint identity mismatch: $($file.FullName)"
    }
    $isDeletion = $deletePaths -contains $file.FullName
    if ($isDeletion) {
        if ($critical.ContainsKey($file.FullName) -or
            $file.FullName.StartsWith($workspaceRoot + '\HRDecoder_Q_20261003\', [System.StringComparison]::OrdinalIgnoreCase) -or
            $file.FullName.StartsWith($workspaceRoot + '\HRDecoder_Ready_20261002\best_pipeline\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Protected checkpoint placed in deletion list: $($file.FullName)"
        }
        if ($record.expected_sha256 -and [string]$record.expected_sha256 -ne $digest) {
            throw "Deletion checkpoint does not match its reviewed evidence: $($file.FullName)"
        }
    }
    # Verify the stat again after hashing, before any deletion can begin.
    $null = Get-CheckedFile $record
    $entry = [ordered]@{
        path = $file.FullName
        bytes = [Int64]$file.Length
        mtime_ns = [Int64]$record.mtime_ns
        sha256_before = $digest
        reason = [string]$record.reason
    }
    if ($isDeletion) { $entry['evidence'] = @($record.evidence); $audit.delete += $entry }
    else { $audit.keep += $entry }
}
foreach ($path in $critical.Keys) {
    if (-not (@($audit.keep.path) -contains $path)) { throw "Protected file absent from retained inventory: $path" }
}
$audit['audit_completed_at_utc'] = [DateTime]::UtcNow.ToString('o')
$audit | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $outDirectory 'predelete_audit.json') -Encoding UTF8
Write-Output "Audit passed: $($audit.delete.Count) deletions, $($audit.keep.Count) retained checkpoint identities."
if (-not $Apply) { return }

# All destructive operations remain in this PowerShell process and use explicit LiteralPath files.
$resultPath = Join-Path $outDirectory 'cleanup_result.json'
$audit.deletion_started = $true
foreach ($entry in $audit.delete) {
    $null = Get-CheckedFile $entry
    $currentHash = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($currentHash -ne $entry.sha256_before) { throw "File changed before deletion: $($entry.path)" }
    Remove-Item -LiteralPath $entry.path -Force
    if (Test-Path -LiteralPath $entry.path) { throw "Deletion did not complete: $($entry.path)" }
    $entry['deleted_at_utc'] = [DateTime]::UtcNow.ToString('o')
    $audit | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resultPath -Encoding UTF8
}
foreach ($entry in $audit.keep) {
    $null = Get-CheckedFile $entry
    $afterHash = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
    $entry['sha256_after'] = $afterHash
    $entry['unchanged'] = $afterHash -eq $entry.sha256_before
    if (-not $entry.unchanged) { throw "Retained checkpoint changed: $($entry.path)" }
}
$audit.complete = $true
$audit['completed_at_utc'] = [DateTime]::UtcNow.ToString('o')
$audit['deleted_count'] = $audit.delete.Count
$deletedBytes = [Int64]0
foreach ($entry in $audit.delete) { $deletedBytes += [Int64]$entry['bytes'] }
$audit['deleted_bytes'] = $deletedBytes
$audit['retained_sha256_verified_count'] = $audit.keep.Count
$audit | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resultPath -Encoding UTF8
Write-Output "Deleted $($audit.deleted_count) checkpoints; bytes released: $($audit.deleted_bytes)."
Write-Output "Verified $($audit.keep.Count) retained checkpoints with unchanged SHA256."

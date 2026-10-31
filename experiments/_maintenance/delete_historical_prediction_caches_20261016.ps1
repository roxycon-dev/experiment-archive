param([switch]$Execute)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$workspaceRoot = (Resolve-Path -LiteralPath 'D:\dataset\experiment').ProviderPath.TrimEnd('\')
$workspacePrefix = $workspaceRoot + '\'
$auditDirectory = [IO.Path]::GetFullPath((Join-Path $workspaceRoot '_maintenance\prediction_cache_cleanup_20261016'))
if (!$auditDirectory.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Audit path outside intended workspace' }

function Get-SafeFiles {
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($workspaceRoot)
    while ($pending.Count) {
        $currentDirectory = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $currentDirectory -Force) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) } else { $item }
        }
    }
}

function Assert-SafeTarget([string]$Path) {
    $resolvedTarget = [IO.Path]::GetFullPath($Path)
    if (!$resolvedTarget.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetExtension($resolvedTarget) -ine '.npz') { throw "Unsafe target: $Path" }
    $ancestor = $resolvedTarget
    while ($ancestor -and $ancestor.Length -ge $workspaceRoot.Length) {
        $item = Get-Item -LiteralPath $ancestor -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point in target: $ancestor" }
        if ($ancestor -ieq $workspaceRoot) { break }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $resolvedTarget
}

function Get-PredictionKind([string]$Path) {
    $relative = $Path.Substring($workspacePrefix.Length).Replace('\','/')
    if ($relative -match '(^|/)(tests?|_shared|gt|labels)(/|$)') { return $null }
    $archive = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $members = @($archive.Entries | ForEach-Object { $_.FullName })
        if ($members.Count -eq 1 -and $members[0] -ceq 'prob.npy') { return 'historical_probability_cache' }
        if ($relative -match '/analysis_results/expert_complementarity_20260920/(contracts|masks)/(fullmodel|hrplus)/' -or
            $relative -match '/analysis_results/expert_complementarity_20260920/logs/smoke_masks/') {
            if ($members -contains 'packed.npy' -and $members -notcontains 'ignore_packed.npy' -and
                @($members | Where-Object { $_ -notin @('packed.npy','n_pixels.npy','orig_hw.npy') }).Count -eq 0) {
                return 'historical_prediction_mask_cache'
            }
        }
        return $null
    } finally { $archive.Dispose() }
}

function Get-ProtectedFingerprint($Files, $TargetSet) {
    $rows = [Collections.Generic.List[string]]::new()
    [long]$bytes = 0
    foreach ($file in $Files) {
        if ($file.FullName.StartsWith($auditDirectory + '\', [StringComparison]::OrdinalIgnoreCase) -or
            $TargetSet.Contains($file.FullName)) { continue }
        $rows.Add($file.FullName + '|' + $file.Length + '|' + $file.LastWriteTimeUtc.Ticks)
        $bytes += $file.Length
    }
    $rows.Sort([StringComparer]::Ordinal)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $hash = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows -join "`n")))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
    return @{ files = $rows.Count; bytes = $bytes; path_size_mtime_sha256 = $hash }
}

function Write-AuditJson([string]$Name, $Value) {
    $target = Join-Path $auditDirectory $Name
    $temporary = $target + '.writing'
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $target -Force
}

if (Test-Path -LiteralPath $auditDirectory) { throw 'Audit folder already exists; inspect existing result before repeating' }
$allFiles = @(Get-SafeFiles)
$targets = [Collections.Generic.List[object]]::new()
$targetSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($file in $allFiles) {
    if ($file.Extension -ine '.npz') { continue }
    $kind = Get-PredictionKind $file.FullName
    if (!$kind) { continue }
    $safeTarget = Assert-SafeTarget $file.FullName
    if (!$targetSet.Add($safeTarget)) { throw "Duplicate target: $safeTarget" }
    $targets.Add(@{ path = $safeTarget; relative_path = $safeTarget.Substring($workspacePrefix.Length);
                    bytes = $file.Length; last_write_utc_ticks = $file.LastWriteTimeUtc.Ticks; kind = $kind })
}
[long]$plannedBytes = ($targets | Measure-Object -Property bytes -Sum).Sum
$protectedBefore = Get-ProtectedFingerprint $allFiles $targetSet
$plan = @{ root = $workspaceRoot; execute = [bool]$Execute; user_request = 'Delete historical prediction/array caches';
           files = $targets.Count; bytes = $plannedBytes; GiB = $plannedBytes / 1GB;
           protected_before = $protectedBefore; targets = @($targets.ToArray());
           preserves = @('original images and labels','model weights','source and configuration','logs and reports',
                         'ground-truth NPZ arrays','synthetic test fixtures','statistical histograms','archives') }
New-Item -ItemType Directory -Path $auditDirectory | Out-Null
Write-AuditJson 'deletion_plan.json' $plan
if (!$Execute) { @{ status = 'PLAN_ONLY'; files = $targets.Count; GiB = $plannedBytes / 1GB; audit = $auditDirectory } | ConvertTo-Json; exit }

# Every candidate is validated again before the first irreversible deletion.
foreach ($row in $targets) {
    $safeTarget = Assert-SafeTarget $row.path
    $current = Get-Item -LiteralPath $safeTarget -Force
    if ($current.Length -ne $row.bytes -or $current.LastWriteTimeUtc.Ticks -ne $row.last_write_utc_ticks -or
        (Get-PredictionKind $safeTarget) -cne $row.kind) { throw "Target changed after inventory: $safeTarget" }
}
$deleted = [Collections.Generic.List[object]]::new()
$failures = [Collections.Generic.List[object]]::new()
[long]$deletedBytes = 0
$diskBefore = ([IO.DriveInfo]::new('D:\')).AvailableFreeSpace
foreach ($row in $targets) {
    try {
        $safeTarget = Assert-SafeTarget $row.path
        Remove-Item -LiteralPath $safeTarget -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $safeTarget) { throw 'Target still exists after deletion' }
        $deleted.Add($row); $deletedBytes += $row.bytes
        @{ path = $row.path; bytes = $row.bytes; status = 'DELETED' } | ConvertTo-Json -Compress |
            Add-Content -LiteralPath (Join-Path $auditDirectory 'deletion_events.jsonl') -Encoding utf8
    } catch {
        $failures.Add(@{ path = $row.path; error = $_.Exception.Message })
        break
    }
}
$remainingFiles = @(Get-SafeFiles)
$protectedAfter = Get-ProtectedFingerprint $remainingFiles $targetSet
$protectedUnchanged = $protectedBefore.path_size_mtime_sha256 -ceq $protectedAfter.path_size_mtime_sha256
$remainingTargets = @($targets | Where-Object { Test-Path -LiteralPath $_.path })
$result = @{ status = if ($failures.Count -eq 0 -and $remainingTargets.Count -eq 0 -and $protectedUnchanged) { 'COMPLETE' } else { 'INCOMPLETE_OR_CHANGED' };
             deleted_files = $deleted.Count; deleted_bytes = $deletedBytes; deleted_GiB = $deletedBytes / 1GB;
             remaining_planned_files = $remainingTargets.Count; failures = @($failures.ToArray());
             protected_files_metadata_unchanged = $protectedUnchanged; protected_before = $protectedBefore;
             protected_after = $protectedAfter; free_space_before_bytes = $diskBefore;
             free_space_after_bytes = ([IO.DriveInfo]::new('D:\')).AvailableFreeSpace; audit_directory = $auditDirectory }
Write-AuditJson 'cleanup_result.json' $result
$result | ConvertTo-Json -Depth 6
if ($result.status -ne 'COMPLETE') { throw 'Cleanup did not fully complete; inspect audit result' }

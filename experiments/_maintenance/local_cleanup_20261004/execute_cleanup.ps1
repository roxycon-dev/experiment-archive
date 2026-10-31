$ErrorActionPreference = 'Stop'
$workspaceRoot = [System.IO.Path]::GetFullPath('D:\dataset\experiment').TrimEnd('\')
$workspacePrefix = $workspaceRoot + '\'
$auditRoot = Join-Path $workspaceRoot '_maintenance\local_cleanup_20261004'
$cleanupPlan = Get-Content -LiteralPath (Join-Path $auditRoot 'delete_plan.json') -Raw | ConvertFrom-Json
$protectedPaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($protectedPath in $cleanupPlan.protected_input_files) { [void]$protectedPaths.Add($protectedPath) }
foreach ($evidence in $cleanupPlan.protected_evidence_files) { [void]$protectedPaths.Add($evidence.path) }
$allowedExtensions = @('.npz', '.pth', '.pt', '.ckpt', '.part', '.download', '.zip', '.tar', '.gz')
$preflightItems = @()
foreach ($entry in $cleanupPlan.delete) {
    $targetPath = [System.IO.Path]::GetFullPath($entry.path)
    if (-not $targetPath.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $targetPath" }
    if ($protectedPaths.Contains($targetPath)) { throw "Protected input/evidence: $targetPath" }
    if ($targetPath.StartsWith((Join-Path $workspaceRoot 'HRDecoder_NextPlan_20261004') + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Current preparation must remain intact: $targetPath" }
    $targetItem = Get-Item -LiteralPath $targetPath -Force
    if ($targetItem.PSIsContainer) { throw "Only explicit files may be removed: $targetPath" }
    if ($targetItem.Extension.ToLowerInvariant() -notin $allowedExtensions) { throw "Disallowed extension: $targetPath" }
    if ($targetItem.Length -ne $entry.bytes) { throw "File changed since planning: $targetPath" }
    $expectedMtimeMs = [long][math]::Floor([double]$entry.mtime_ns / 1000000)
    $actualMtimeMs = ([DateTimeOffset]$targetItem.LastWriteTimeUtc).ToUnixTimeMilliseconds()
    if ([math]::Abs($expectedMtimeMs - $actualMtimeMs) -gt 1) { throw "Modification time changed: $targetPath" }
    # File-only deletion, with every ancestor checked for redirects/junctions.
    $pathToCheck = $targetPath
    while ($pathToCheck.Length -ge $workspaceRoot.Length) {
        $ancestor = Get-Item -LiteralPath $pathToCheck -Force
        if (($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Redirected path rejected: $pathToCheck" }
        if ($pathToCheck.Equals($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $pathToCheck = [System.IO.Path]::GetDirectoryName($pathToCheck)
    }
    $preflightItems += [pscustomobject]@{ path = $targetPath; bytes = [long]$entry.bytes; reason = $entry.reason }
}
$receipt = [ordered]@{
    stage = 'DELETING_AUTHORIZED_UNUSED_DATA_ONLY'
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    root = $workspaceRoot
    removed = [System.Collections.Generic.List[object]]::new()
    deleted_bytes = [long]0
    experiment_scripts_and_logs_protected = $true
    recursive_directory_deletion = $false
}
$receiptPath = Join-Path $auditRoot 'cleanup_result.json'
foreach ($entry in $preflightItems) {
    Remove-Item -LiteralPath $entry.path -Force -ErrorAction Stop
    $receipt.removed.Add($entry)
    $receipt.deleted_bytes += $entry.bytes
    if ($receipt.removed.Count % 100 -eq 0) {
        $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
        Write-Host "Removed $($receipt.removed.Count) files; $([math]::Round($receipt.deleted_bytes / 1GB, 2)) GiB"
    }
}
foreach ($evidence in $cleanupPlan.protected_evidence_files) {
    if (-not [System.IO.File]::Exists($evidence.path)) { throw "Missing script/log/evidence: $($evidence.path)" }
    if ((Get-Item -LiteralPath $evidence.path).Length -ne $evidence.bytes) { throw "Script/log/evidence size changed: $($evidence.path)" }
}
foreach ($protectedPath in $cleanupPlan.protected_input_files) {
    if (-not [System.IO.File]::Exists($protectedPath)) { throw "Missing protected experiment input: $protectedPath" }
}
$receipt.stage = 'COMPLETE'
$receipt.completed_at_utc = [DateTime]::UtcNow.ToString('o')
$receipt.verified_preserved_evidence_files = $cleanupPlan.protected_evidence_files.Count
$receipt.verified_preserved_input_files = $cleanupPlan.protected_input_files.Count
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Host "Complete: removed $($receipt.removed.Count) files; freed $([math]::Round($receipt.deleted_bytes / 1GB, 2)) GiB. All scripts/logs retained."

$ErrorActionPreference = 'Stop'
$planPath = 'D:\dataset\experiment\_maintenance\next_retired_U1_cleanup_plan_20261004.json'
$resultPath = 'D:\dataset\experiment\_maintenance\next_retired_U1_cleanup_result_20261004.json'
$policyPath = 'D:\dataset\experiment\_maintenance\deferred_cleanup_policy_20261004.json'
$campaignRoot = 'D:\dataset\experiment\HRDecoder_Q_20261003'
$intendedRoot = 'D:\dataset\experiment\HRDecoder_Next_20261003'
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Save-Json($path, $value) {
    [System.IO.File]::WriteAllText($path, ($value | ConvertTo-Json -Depth 100), $utf8)
}
function Assert-True($condition, $message) {
    if (-not $condition) { throw $message }
}
function Read-Json($path) { Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json }
function Check-Sha($path, $expected) {
    Assert-True ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $expected) "SHA differs: $path"
}
Assert-True (-not (Test-Path -LiteralPath $resultPath)) 'Existing cleanup result requires review; no repeated deletion'
$plan = Read-Json $planPath
Assert-True ($plan.stage -eq 'REVIEWED_EXACT_ALLOWLIST_READY') 'Reviewed allowlist required'
$root = (Resolve-Path -LiteralPath $intendedRoot).Path.TrimEnd('\')
Assert-True ($root -eq $plan.root) 'Resolved cleanup root differs'
Assert-True ($plan.original_plan_experiments_and_evaluation_complete -and $plan.necessary_local_results_SHA_verified) 'Completion and backup gates required'
Check-Sha $policyPath $plan.policy_sha256
Check-Sha "$campaignRoot\reports\original_plan_report_receipt.json" $plan.report_receipt_sha256
Check-Sha "$campaignRoot\server_results\continuation_evidence\reports\continuation_final_selection.json" $plan.selected_lock_sha256
foreach ($run in @('Q_DDR_seed42','Q_DDR_seed2024','Q_MIX_seed42','Q_MIX_seed2024')) {
    $backup = Read-Json "$campaignRoot\reports\run_backup_${run}_24k.json"
    Assert-True ($backup.complete -and $backup.files.Count -eq 327) "Incomplete local backup: $run"
}
foreach ($row in $plan.files) {
    $resolved = (Resolve-Path -LiteralPath $row.path).Path
    $parent = [System.IO.Path]::GetDirectoryName($resolved)
    Assert-True ($resolved -eq $row.path -and $resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) 'Deletion target escapes intended root'
    Assert-True ($plan.allowed_parent_directories -contains $parent) 'Deletion parent absent from explicit allowlist'
    Assert-True ([System.IO.Path]::GetExtension($resolved) -eq '.npz') 'Only retired probability NPZ may be deleted'
    $ancestor = $resolved
    while ($ancestor -and $ancestor.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        $item = Get-Item -LiteralPath $ancestor
        Assert-True (-not ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) 'Reparse point target forbidden'
        if ($ancestor -eq $root) { break }
        $ancestor = [System.IO.Path]::GetDirectoryName($ancestor)
    }
    Assert-True ((Get-Item -LiteralPath $resolved).Length -eq $row.bytes) 'Deletion target size differs'
    Check-Sha $resolved $row.sha256
}
foreach ($row in $plan.preserved_metadata) { Check-Sha $row.path $row.sha256 }
Check-Sha $plan.preserved_U1_model.path $plan.preserved_U1_model.sha256
$deleted = New-Object System.Collections.ArrayList
$receipt = [ordered]@{stage='ALL_TARGETS_REVALIDATED_DELETION_STARTED'; started_at=[DateTime]::UtcNow.ToString('o'); plan_path=$planPath; plan_sha256=(Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant(); deleted_files=$deleted; deleted_bytes=0; recursive_delete_performed=$false; checkpoint_deletion_performed=$false}
Save-Json $resultPath $receipt
try {
    foreach ($row in $plan.files) {
        Remove-Item -LiteralPath $row.path
        [void]$deleted.Add($row)
        $receipt.deleted_bytes += [long]$row.bytes
        Save-Json $resultPath $receipt
    }
    foreach ($row in $plan.preserved_model_files) { Assert-True ((Get-Item -LiteralPath $row.path).Length -eq $row.bytes) 'Protected model changed' }
    Check-Sha $plan.preserved_U1_model.path $plan.preserved_U1_model.sha256
    foreach ($row in $plan.preserved_metadata) { Check-Sha $row.path $row.sha256 }
    foreach ($cache in $plan.protected_baseline_caches) {
        $actual = @(Get-ChildItem -LiteralPath $cache.path -Filter '*.npz' -File)
        Assert-True ($actual.Count -eq $cache.files.Count) 'Protected baseline cache file count changed'
        foreach ($row in $cache.files) { Assert-True ((Get-Item -LiteralPath (Join-Path $cache.path $row.name)).Length -eq $row.bytes) 'Protected baseline cache changed' }
    }
    $receipt.stage = 'COMPLETE_EXACT_ALLOWLIST_DELETION_PROTECTED_ASSETS_VERIFIED'
    $receipt.completed_at = [DateTime]::UtcNow.ToString('o')
    $receipt.protected_models = $plan.preserved_model_files.Count
    $receipt.protected_metadata_SHA_verified = $true
    $receipt.protected_baseline_caches_verified = $true
    Save-Json $resultPath $receipt
    $policy = Read-Json $policyPath
    $policy | Add-Member -NotePropertyName further_cleanup_started -NotePropertyValue $true -Force
    $policy | Add-Member -NotePropertyName further_cleanup_complete -NotePropertyValue $true -Force
    $policy | Add-Member -NotePropertyName cleanup_receipt -NotePropertyValue $resultPath -Force
    $policy | Add-Member -NotePropertyName cleanup_completed_at -NotePropertyValue $receipt.completed_at -Force
    Save-Json $policyPath $policy
    Write-Output "COMPLETE: $($deleted.Count) retired NPZ files, $($receipt.deleted_bytes) bytes; all 11 models and protected evidence retained"
} catch {
    $receipt.stage = 'TECHNICAL_PAUSE_NO_AUTOMATIC_RETRY'
    $receipt.error = $_.Exception.Message
    Save-Json $resultPath $receipt
    throw
}

param([string]$ReportRoot = 'D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告')
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $inputPath = Join-Path $ReportRoot 'HRDecoder项目结题报告.docx'
    $pdfPath = Join-Path $ReportRoot 'HRDecoder项目结题报告.pdf'
    $document = $word.Documents.Open($inputPath, $false, $false)
    $null = $document.Fields.Update()
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    $document.Save()
    $document.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    [ordered]@{ renderer='Microsoft Word native COM hidden isolated authoring instance'; pages=$pages; pdf=$pdfPath; word_version=$word.Version; generated_at_utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ReportRoot 'evidence\native_word_export_receipt.json') -Encoding utf8
    Write-Output "Word export completed: $pages pages"
}
finally {
    if ($null -ne $document) {
        try { $document.Close(0) } catch { Write-Verbose "The isolated document instance already closed: $($_.Exception.Message)" }
        try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) } catch { }
    }
    if ($null -ne $word) {
        try { $word.Quit(0) } catch { Write-Verbose "The isolated Word instance already exited: $($_.Exception.Message)" }
        try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) } catch { }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

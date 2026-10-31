$ErrorActionPreference='Stop'
$reportRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$reportTitle='HRDecoder最佳模型训练与改进报告'
$word=$null
$document=$null
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $word.AutomationSecurity=3
    $document=$word.Documents.Open((Join-Path $reportRoot ($reportTitle+'.docx')),$false,$false)
    $null=$document.Fields.Update()
    $document.Repaginate()
    $pages=$document.ComputeStatistics(2)
    $document.Save()
    $pdfPath=Join-Path $reportRoot ($reportTitle+'.pdf')
    $document.ExportAsFixedFormat($pdfPath,17,$false,0,0,1,1,0,$true,$true,1,$true,$true,$false)
    [ordered]@{renderer='Microsoft Word';pages=$pages;generated_at=[DateTime]::UtcNow.ToString('o');pdf=$pdfPath} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportRoot 'PDF导出记录.json') -Encoding UTF8
    Write-Output "PDF exported: $pages pages"
}
finally {
    if ($null -ne $document) { try {$document.Close(0)} catch {};try {[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)} catch {} }
    if ($null -ne $word) { try {$word.Quit(0)} catch {};try {[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)} catch {} }
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
}

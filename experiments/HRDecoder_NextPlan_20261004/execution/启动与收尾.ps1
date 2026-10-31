param([switch]$Launch)
$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$experimentPython = 'C:\Users\GuanXingRen\Miniconda3\python.exe'
Set-Location -LiteralPath $executionRoot
if ($Launch) {
    & $experimentPython -u 'ops\deploy.py' --launch
    if ($LASTEXITCODE -ne 0) { throw '部署或启动失败；查看reports，禁止盲目重试。' }
} else {
    Write-Host '本地准备已完成。用户开启服务器后，运行本脚本并传入 -Launch。'
    Write-Host '45分钟巡检已创建；部署启动时会同步开启隐藏的成果回收与关机收尾进程。'
}

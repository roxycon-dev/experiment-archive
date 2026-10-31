param(
    [string]$DataRoot,
    [string]$Pretrained,
    [string]$Identity = "$PSScriptRoot/preparation_20261016/data_identity.json",
    [string]$Output = "$PSScriptRoot/preparation_20261016/static_checks",
    [switch]$ServerPreflight
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($ServerPreflight) {
        if (!$DataRoot -or !$Pretrained) { throw 'ServerPreflight needs DataRoot and Pretrained.' }
        $env:NATIVE_PYRAMID_PRETRAINED = $Pretrained
        python verify.py --data-root $DataRoot --identity $Identity --pretrained $Pretrained --output $Output --crop-steps 256
    } else {
        python verify.py --static-only --output $Output
    }
    if ($LASTEXITCODE -ne 0) { throw "Verification failed with exit code $LASTEXITCODE" }
    if ($ServerPreflight) {
        $check = Get-Content -LiteralPath "$Output/preflight.json" -Raw | ConvertFrom-Json
        if ($check.status -ne 'PASS') { throw "Preflight did not pass: $($check.status)" }
    }
} finally {
    Pop-Location
}

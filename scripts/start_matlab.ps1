$ErrorActionPreference = 'Stop'
$matlabExe = Join-Path $env:USERPROFILE 'MATLAB\R2026a\bin\matlab.exe'
if (-not (Test-Path -LiteralPath $matlabExe)) { throw "MATLAB not found: $matlabExe" }
$projectRoot = Split-Path -Parent $PSScriptRoot
Start-Process -FilePath $matlabExe -ArgumentList @('-licmode', 'onlinelicensing', '-sd', ('"' + $projectRoot + '"'))

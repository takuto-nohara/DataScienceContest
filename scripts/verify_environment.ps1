$ErrorActionPreference = 'Stop'
$matlabExe = Join-Path $env:USERPROFILE 'MATLAB\R2026a\bin\matlab.exe'
if (-not (Test-Path -LiteralPath $matlabExe)) { throw "MATLAB not found: $matlabExe" }
$checkFile = (Join-Path $PSScriptRoot 'verify_environment.m').Replace("'", "''")
& $matlabExe -wait -licmode onlinelicensing -batch "run('$checkFile')"
if ($LASTEXITCODE -ne 0) { throw "MATLAB verification failed ($LASTEXITCODE)." }

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'native_utf8.ps1')
$matlabExe = Join-Path $env:USERPROFILE 'MATLAB\R2026a\bin\matlab.exe'
if (-not (Test-Path -LiteralPath $matlabExe)) { throw "MATLAB not found: $matlabExe" }
$checkFile = (Join-Path $PSScriptRoot 'verify_environment.m').Replace("'", "''")
$nativeExitCode = Invoke-Utf8NativeProcess -FilePath $matlabExe -ArgumentList @('-wait','-licmode','onlinelicensing','-batch',"run('$checkFile')") -LogPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts\environment-output.log')
if ($nativeExitCode -ne 0) { throw "MATLAB verification failed ($nativeExitCode)." }
exit $nativeExitCode

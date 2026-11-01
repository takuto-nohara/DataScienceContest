param([ValidateSet('all','train','evaluate','test')][string]$Action='all', [string[]]$Models=@(), [string]$OutputName='latest')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$matlabExe=Join-Path $env:USERPROFILE 'MATLAB\R2026a\bin\matlab.exe'
foreach ($modelId in $Models) { if ($modelId -notmatch '^proto_[A-Za-z0-9_]+$') { throw "Invalid model ID: $modelId" } }
if ($OutputName -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid output name.' }
$rootLiteral=$projectRoot.Replace("'","''").Replace('\','/')
$modelLiteral='{' + (($Models | ForEach-Object { "'"+$_+"'" }) -join ',') + '}'
if ($Models.Count -eq 0) { $modelLiteral='string.empty' }
$command="addpath('$rootLiteral/matlab');"
if ($Action -eq 'test') { $command+="run('$rootLiteral/tests/run_checks.m');" }
if ($Action -in @('all','train')) { $command+="har.train('$rootLiteral',$modelLiteral);" }
if ($Action -in @('all','evaluate')) { $command+="har.evaluate('$rootLiteral',$modelLiteral,'$OutputName');" }
& $matlabExe -wait -licmode onlinelicensing -batch $command
if ($LASTEXITCODE -ne 0) { throw "Prototype $Action failed ($LASTEXITCODE)." }

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'scripts\native_utf8.ps1')
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$expectedArguments = @('path with spaces', 'a"b', 'trailing\', 'backslash\"quote')
$argumentLog = Join-Path $projectRoot 'artifacts\argument-roundtrip.log'
$arguments = @('-X','utf8','-c','import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(7)') + $expectedArguments
$nativeExitCode = Invoke-Utf8NativeProcess -FilePath $pythonExe -ArgumentList $arguments -NativeTextEncoding ([Text.Encoding]::UTF8) -LogPath $argumentLog
if ($nativeExitCode -ne 7) { throw 'Native nonzero exit code was lost.' }
$actualArguments = Get-Content -LiteralPath $argumentLog -Raw -Encoding UTF8 | ConvertFrom-Json
if (Compare-Object $expectedArguments $actualArguments -SyncWindow 0) { throw 'Native argument quoting failed.' }
$matlabExe = Join-Path $env:USERPROFILE 'MATLAB\R2026a\bin\matlab.exe'
$rootLiteral = $projectRoot.Replace("'","''").Replace('\','/')
$log = Join-Path $projectRoot 'artifacts\encoding-validation.log'
$nativeExitCode = Invoke-Utf8NativeProcess -FilePath $matlabExe -ArgumentList @('-wait','-licmode','onlinelicensing','-batch',"run('$rootLiteral/tests/check_encoding.m');") -LogPath $log
if ($nativeExitCode -ne 0) { throw 'MATLAB encoding validation failed.' }
$expectedJapanese = -join (@(26085,26412,35486,12398,34920,31034,30906,35469) | ForEach-Object { [char]$_ })
$loggedText = [IO.File]::ReadAllText($log, [Text.Encoding]::UTF8)
if (-not $loggedText.Contains($expectedJapanese)) { throw 'Japanese output was not preserved in the UTF-8 log.' }
Write-Output 'NATIVE_UTF8_CHECK_PASSED: argument quoting, exit codes, Japanese MATLAB output and UTF-8 log'

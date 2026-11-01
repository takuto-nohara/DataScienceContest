# Download only from MathWorks and verify the executable before running it.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'native_utf8.ps1')
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this installation script in an administrator PowerShell. MathWorks OS registration requires elevation.'
}
$setupDir = Join-Path $env:LOCALAPPDATA 'MathWorks\ContestSetup'
New-Item -ItemType Directory -Force -Path $setupDir | Out-Null
$mpmExe = Join-Path $setupDir 'mpm.exe'
Invoke-WebRequest -Uri 'https://www.mathworks.com/mpm/win64/mpm' -OutFile $mpmExe
$signature = Get-AuthenticodeSignature -LiteralPath $mpmExe
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*O="The MathWorks, Inc."*') {
    throw 'The MathWorks signature could not be verified.'
}
$destination = Join-Path $env:USERPROFILE 'MATLAB\R2026a'
$nativeExitCode = Invoke-Utf8NativeProcess -FilePath $mpmExe -ArgumentList @('install','--release=R2026aU5',"--destination=$destination",'--products=MATLAB','Deep_Learning_Toolbox','Parallel_Computing_Toolbox','Statistics_and_Machine_Learning_Toolbox','Signal_Processing_Toolbox') -NativeTextEncoding ([System.Text.Encoding]::UTF8) -LogPath (Join-Path $setupDir 'installation-utf8.log')
if ($nativeExitCode -ne 0) { throw "MATLAB installation failed ($nativeExitCode)." }
exit $nativeExitCode

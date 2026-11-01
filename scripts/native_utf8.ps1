# Decode native Japanese-Windows output before PowerShell emits Unicode text.
# This does not change Windows locale, global console settings, or model files.
function ConvertTo-NativeArgument {
    param([string]$Value)
    $quoted = New-Object System.Text.StringBuilder
    [void]$quoted.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            [void]$quoted.Append(('\' * (2 * $slashes + 1)))
            [void]$quoted.Append('"')
        } else {
            [void]$quoted.Append(('\' * $slashes))
            [void]$quoted.Append($character)
        }
        $slashes = 0
    }
    [void]$quoted.Append(('\' * (2 * $slashes)))
    [void]$quoted.Append('"')
    return $quoted.ToString()
}

function Invoke-Utf8NativeProcess {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(Mandatory=$true)][string[]]$ArgumentList,
        [string]$LogPath,
        [System.Text.Encoding]$NativeTextEncoding = [System.Text.Encoding]::GetEncoding([Globalization.CultureInfo]::CurrentCulture.TextInfo.ANSICodePage)
    )
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $FilePath
    $info.Arguments = ($ArgumentList | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
    $info.WorkingDirectory = (Get-Location).Path
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = $NativeTextEncoding
    $info.StandardErrorEncoding = $NativeTextEncoding
    $writer = $null
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        if ($LogPath) {
            $folder = Split-Path -Parent $LogPath
            if ($folder -and -not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
            $writer = New-Object System.IO.StreamWriter($LogPath, $false, (New-Object System.Text.UTF8Encoding($false)))
        }
        [void]$process.Start()
        # Drain stderr concurrently to avoid a full stderr pipe blocking stdout.
        $errorTask = $process.StandardError.ReadToEndAsync()
        while ($null -ne ($line = $process.StandardOutput.ReadLine())) {
            Write-Host $line
            if ($writer) { $writer.WriteLine($line); $writer.Flush() }
        }
        $process.WaitForExit()
        $errorText = $errorTask.GetAwaiter().GetResult()
        if ($errorText) {
            Write-Host $errorText
            if ($writer) { $writer.Write($errorText) }
        }
        return $process.ExitCode
    } finally {
        if ($writer) { $writer.Dispose() }
        $process.Dispose()
    }
}

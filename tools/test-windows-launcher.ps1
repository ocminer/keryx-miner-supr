param([Parameter(Mandatory = $true)][string]$Launcher)
$ErrorActionPreference = 'Stop'

# Exercise the packaged launcher without starting mining. A native test executable
# records its working directory and exits with a distinctive nonzero status.
$root = Join-Path ([System.IO.Path]::GetTempPath()) ('keryx-launcher-' + [guid]::NewGuid())
$package = Join-Path $root 'package with spaces'
$foreign = Join-Path $root 'different working directory'
New-Item -ItemType Directory -Path $package, $foreign | Out-Null
$previousLog = $env:KERYX_LAUNCH_TEST_LOG
try {
    Copy-Item $Launcher (Join-Path $package 'start.bat')
    $env:KERYX_LAUNCH_TEST_LOG = Join-Path $root 'working-directory.txt'
    $stub = Join-Path $root 'stub.cs'
    @'
using System;
using System.IO;
class LauncherProbe {
    static int Main() {
        File.WriteAllText(Environment.GetEnvironmentVariable("KERYX_LAUNCH_TEST_LOG"),
                          Environment.CurrentDirectory);
        return 37;
    }
}
'@ | Set-Content -Encoding ascii $stub
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $compiler /nologo /target:exe "/out:$package\keryx-miner-supr.exe" $stub
    if ($LASTEXITCODE -ne 0) { throw 'Launcher probe compilation failed' }
    $command = '""' + (Join-Path $package 'start.bat') + '" <NUL"'
    $process = Start-Process -FilePath $env:ComSpec -ArgumentList '/d', '/c', $command `
        -WorkingDirectory $foreign -NoNewWindow -PassThru
    if (-not $process.WaitForExit(30000)) {
        $process.Kill()
        throw 'Launcher did not exit within 30 seconds'
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 37) { throw "Launcher lost exit status: $($process.ExitCode)" }
    $actual = (Get-Content -Raw $env:KERYX_LAUNCH_TEST_LOG).TrimEnd('\')
    if ($actual -ne $package.TrimEnd('\')) { throw "Wrong miner working directory: $actual" }
    Write-Host 'Windows AMD launcher foreign-directory and exit-status test PASS'
} finally {
    $env:KERYX_LAUNCH_TEST_LOG = $previousLog
    Remove-Item -LiteralPath $root -Recurse -Force
}

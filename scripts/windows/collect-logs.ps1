#Requires -Version 7.2
[CmdletBinding()]
param([switch]$Screenshot, [switch]$Zip)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
$destination = Join-Path $ProjectRoot ('.work/diagnostics/vm-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $destination | Out-Null
$hostReport = [ordered]@{
    collected_utc = [DateTime]::UtcNow.ToString('o'); powershell = $PSVersionTable.PSVersion.ToString()
    whpx = Get-ReimsWhpxStatus
    os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
    cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
    gpu = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
}
Write-ReimsJson (Join-Path $destination 'host.json') $hostReport
foreach ($file in @('config.json', 'media.json', 'opencore.json', 'runtime.json')) {
    $source = Join-Path $VmRoot "state/$file"
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $destination }
}
$runtime = Get-ReimsRuntime
if ($runtime) {
    [void](Assert-ReimsArtifactPath $runtime.run_directory)
    foreach ($name in @('stdout.log', 'stderr.log', 'serial.log', 'command.json', 'qmp-status.json')) {
        $source = Join-Path $runtime.run_directory $name
        if (Test-Path -LiteralPath $source) {
            # QEMU keeps logs open while running; allow the writer to continue.
            $inputFile = [IO.File]::Open($source, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
            $outputFile = [IO.File]::Create((Join-Path $destination $name))
            try { $inputFile.CopyTo($outputFile) } finally { $inputFile.Dispose(); $outputFile.Dispose() }
        }
    }
    if (Get-ReimsVmProcess $runtime) {
        try {
            Write-ReimsJson (Join-Path $destination 'qmp-status.json') (Invoke-ReimsQmp $runtime 'query-status')
            Write-ReimsJson (Join-Path $destination 'qmp-block.json') (Invoke-ReimsQmp $runtime 'query-block')
            if ($Screenshot) {
                Invoke-ReimsQmp $runtime 'screendump' @{ filename = (Join-Path $destination 'guest.ppm'); device = 'gfx'; head = 0 } | Out-Null
            }
        } catch { $_.Exception.Message | Set-Content -LiteralPath (Join-Path $destination 'qmp-error.txt') -Encoding utf8NoBOM }
    } elseif ($Screenshot) { 'VM is stopped; no guest screendump collected.' | Set-Content -LiteralPath (Join-Path $destination 'qmp-error.txt') }
} elseif ($Screenshot) { 'No VM runtime recorded; no guest screendump collected.' | Set-Content -LiteralPath (Join-Path $destination 'qmp-error.txt') }
$stderr = Join-Path $destination 'stderr.log'
if (Test-Path -LiteralPath $stderr) {
    $failures = Select-String -LiteralPath $stderr -Pattern 'reims-vgpu: fatal|panic|WHPX.*fail|VK_ERROR|assertion' -CaseSensitive:$false
    $failures | ForEach-Object Line | Set-Content -LiteralPath (Join-Path $destination 'failures.txt') -Encoding utf8NoBOM
}
if ($Zip) { Compress-Archive -LiteralPath $destination -DestinationPath "$destination.zip" }
"Local diagnostics: $destination. Review logs and host paths before sharing."

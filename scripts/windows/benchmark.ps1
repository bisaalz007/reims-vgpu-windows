#Requires -Version 7.2
[CmdletBinding()]
param([ValidateRange(1, 3600)][int]$Seconds = 60, [string]$GuestUser, [string]$IdentityFile)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
if ($GuestUser -and $GuestUser -notmatch '^[A-Za-z_][A-Za-z0-9_.-]*$') { throw 'Invalid guest user name.' }
$runtime = Get-ReimsRuntime
if (!$runtime -or !(Get-ReimsVmProcess $runtime)) { throw 'Start the VM before sampling it.' }
if ($runtime.diagnostic_only) { throw 'A paused TCG diagnostic is not a performance benchmark.' }
$config = Get-ReimsVmConfig
$status = Invoke-ReimsQmp $runtime 'query-status'
if (!$status.running) { throw 'VM is paused; no running-VM benchmark can be collected.' }
$directory = Join-Path $ProjectRoot ('.work/diagnostics/benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $directory | Out-Null
$samples = [Collections.Generic.List[object]]::new()
$stderr = Join-Path $runtime.run_directory 'stderr.log'
$logOffset = if (Test-Path -LiteralPath $stderr) { (Get-Item -LiteralPath $stderr).Length } else { 0 }
$clock = [Diagnostics.Stopwatch]::StartNew()
$previous = (Get-ReimsVmProcess $runtime).TotalProcessorTime.TotalSeconds
$previousTime = 0.0
while ($clock.Elapsed.TotalSeconds -lt $Seconds) {
    Start-Sleep -Milliseconds 1000
    $process = Get-ReimsVmProcess $runtime
    if (!$process) { throw "VM exited during benchmark. Partial diagnostics: $directory" }
    $elapsed = $clock.Elapsed.TotalSeconds
    $cpu = $process.TotalProcessorTime.TotalSeconds
    $samples.Add([pscustomobject]@{
        elapsed_seconds = [math]::Round($elapsed, 3)
        cpu_core_percent = [math]::Round(100 * ($cpu - $previous) / ($elapsed - $previousTime), 2)
        working_set_bytes = $process.WorkingSet64; private_bytes = $process.PrivateMemorySize64
    })
    $samples[-1] | Export-Csv -LiteralPath (Join-Path $directory 'host-process.csv') -Append -NoTypeInformation
    $previous = $cpu
    $previousTime = $elapsed
}
$summary = [ordered]@{
    schema = 1; seconds = $clock.Elapsed.TotalSeconds; sample_count = $samples.Count; accelerator = 'whpx'
    metric = 'host QEMU process CPU and memory during the user workload; 100% CPU is one logical core'
    average_cpu_core_percent = ($samples | Measure-Object cpu_core_percent -Average).Average
    peak_working_set_bytes = ($samples | Measure-Object working_set_bytes -Maximum).Maximum
    guest_probe = 'not requested'; run_directory = $runtime.run_directory
    display_stats = 'no samples reported during this interval'
}
if (Test-Path -LiteralPath $stderr) {
    $file = [IO.File]::Open($stderr, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        [void]$file.Seek($logOffset, [IO.SeekOrigin]::Begin)
        $reader = [IO.StreamReader]::new($file)
        $stats = @($reader.ReadToEnd() -split "`r?`n" | Where-Object { $_ -match '^reims-vgpu stats:' })
        if ($stats.Count) {
            $stats | Set-Content -LiteralPath (Join-Path $directory 'display-stats.log') -Encoding utf8NoBOM
            $summary.display_stats = 'raw per-second guest presents, window draws and window GPU times in display-stats.log'
        }
    } finally { $file.Dispose() }
}
if ($GuestUser) {
    $ssh = Get-ReimsTool 'ssh.exe'
    $sshArgs = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
        '-o', ('UserKnownHostsFile=' + (Join-Path $VmRoot 'state/known_hosts')), '-p', "$($config.ssh_port)")
    if ($IdentityFile) { $sshArgs += '-i', $IdentityFile }
    $sshArgs += "$GuestUser@127.0.0.1", 'sw_vers; uname -m; sysctl hw.ncpu hw.memsize; system_profiler SPDisplaysDataType'
    $probe = Invoke-ReimsTool $ssh $sshArgs
    $probe | Set-Content -LiteralPath (Join-Path $directory 'guest-info.txt') -Encoding utf8NoBOM
    $summary.guest_probe = 'read-only inventory over an existing SSH key and known host'
}
Write-ReimsJson (Join-Path $directory 'summary.json') $summary
"Host process samples: $directory. This does not measure guest GPU throughput or prove screen correctness."

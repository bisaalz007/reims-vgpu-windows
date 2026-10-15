#Requires -Version 7.2
[CmdletBinding()]
param([ValidateRange(1, 300)][int]$TimeoutSeconds = 60, [switch]$Exit, [switch]$Force)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
$runtime = Get-ReimsRuntime
if (!$runtime -or !(Get-ReimsVmProcess $runtime)) { 'The recorded VM is stopped.'; return }
try {
    $quitRequested = [bool]$runtime.diagnostic_only
    if ($quitRequested) {
        Invoke-ReimsQmp $runtime 'quit' | Out-Null
    } elseif ((Invoke-ReimsQmp $runtime 'query-status').status -ne 'shutdown') {
        Invoke-ReimsQmp $runtime 'system_powerdown' | Out-Null
    }
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if (!(Get-ReimsVmProcess $runtime)) { 'QEMU stopped.'; return }
        if (!$quitRequested -and (Invoke-ReimsQmp $runtime 'query-status').status -eq 'shutdown') {
            if (!$Exit) { "Guest powered off; QEMU PID $($runtime.pid) retained. Run start.ps1 to power on."; return }
            Invoke-ReimsQmp $runtime 'quit' | Out-Null
            $quitRequested = $true
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'Guest has not shut down. Use stop.ps1 -Force only to discard unsaved work and exit QEMU.'
} catch {
    if (!$Force) { throw }
    Write-Warning $_.Exception.Message
}
try { Invoke-ReimsQmp $runtime 'quit' | Out-Null } catch { Write-Warning $_.Exception.Message }
Start-Sleep -Milliseconds 500
$owned = Get-ReimsVmProcess $runtime
if ($owned) { Stop-Process -InputObject $owned -Force }
'Powered off the recorded VM.'

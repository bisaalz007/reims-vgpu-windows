# Run manually in an administrator PowerShell. This script never restarts Windows.
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator PowerShell is required to enable Windows Hypervisor Platform.'
}
Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -All -NoRestart
Write-Output 'Save your work and restart Windows before starting the WHPX VM.'

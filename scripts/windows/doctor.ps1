param([switch]$RequireWhpx)
. (Join-Path $PSScriptRoot 'environment.ps1')
if (-not ('ReimsWhpx' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ReimsWhpx {
    [DllImport("WinHvPlatform.dll")]
    private static extern int WHvGetCapability(uint code, out ulong value, uint size, out uint written);
    public static bool Present(out int result) {
        ulong value; uint written;
        result = WHvGetCapability(0, out value, 8, out written);
        return result >= 0 && written >= 4 && (value & 0xffffffffUL) != 0;
    }
}
'@
}
$capabilityResult = 0
try { $whpxPresent = [ReimsWhpx]::Present([ref]$capabilityResult) }
catch { $whpxPresent = $false; $capabilityResult = $_.Exception.HResult }
$feature = Get-CimInstance Win32_OptionalFeature -Filter "Name = 'HypervisorPlatform'"
$cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, VirtualizationFirmwareEnabled
$leaves = @()
if ([System.Runtime.Intrinsics.X86.X86Base]::IsSupported) {
    foreach ($leaf in @(2, 4, 5, 11, 31)) {
        $registers = [System.Runtime.Intrinsics.X86.X86Base]::CpuId($leaf, 0)
        $leaves += [ordered]@{ Leaf = $leaf; EAX = $registers.Item1; EBX = $registers.Item2; ECX = $registers.Item3; EDX = $registers.Item4 }
    }
}
$tools = [ordered]@{}
foreach ($name in @('git', 'rustup', 'cargo', 'rustc', 'clang', 'lld-link', 'llvm-objcopy', 'cmake', 'ninja', 'pkg-config', 'python')) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    $tools[$name] = if ($command) { $command.Source } else { $null }
}
$report = [ordered]@{
    OS = (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, TotalVisibleMemorySize, FreePhysicalMemory)
    CPU = $cpu
    CPUID = $leaves
    GPU = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    WhpxFeatureEnabled = ($feature.InstallState -eq 1)
    WhpxHypervisorPresent = $whpxPresent
    WhpxCapabilityResult = $capabilityResult
    HypervisorPresent = (Get-CimInstance Win32_ComputerSystem).HypervisorPresent
    FreeBytes = (Get-PSDrive -Name ([IO.Path]::GetPathRoot($ProjectRoot).Substring(0, 1))).Free
    Tools = $tools
}
$json = $report | ConvertTo-Json -Depth 6
$json | Set-Content -LiteralPath (Join-Path $ProjectRoot '.work\diagnostics\doctor.json') -Encoding UTF8
$json
if ($RequireWhpx -and (-not $report.WhpxFeatureEnabled -or -not $report.WhpxHypervisorPresent)) {
    throw 'WHPX is unavailable. Enable HypervisorPlatform from an administrator PowerShell and restart Windows; TCG is not selected.'
}

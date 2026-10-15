#Requires -Version 7.2
[CmdletBinding()]
param(
    [ValidateSet('Create', 'Restore', 'List')][string]$Action = 'List',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')][string]$Name,
    [string]$QemuImg
)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
$snapshots = Join-Path $VmRoot 'snapshots'
if ($Action -eq 'List') {
    if (Test-Path -LiteralPath $snapshots) {
        Get-ChildItem -LiteralPath $snapshots -Directory | ForEach-Object {
            $file = Join-Path $_.FullName 'snapshot.json'
            if (Test-Path -LiteralPath $file) { Get-Content -Raw -LiteralPath $file | ConvertFrom-Json }
        }
    }
    return
}
if (!$Name) { throw 'Pass -Name for Create or Restore.' }
Assert-ReimsVmStopped
$config = Get-ReimsVmConfig
$QemuImg = Get-ReimsTool 'qemu-img.exe' $QemuImg
$directory = Join-Path $snapshots $Name
$manifest = Join-Path $directory 'snapshot.json'
$next = Join-Path $VmRoot ("disks/" + [guid]::NewGuid().ToString() + '.qcow2')
if ($Action -eq 'Create') {
    if (Test-Path -LiteralPath $directory) { throw "Snapshot already exists: $Name" }
    Invoke-ReimsTool $QemuImg @('check', '-q', $config.active_disk) | Out-Null
    Invoke-ReimsTool $QemuImg @('create', '-f', 'qcow2', '-F', 'qcow2', '-b', $config.active_disk, $next) | Out-Null
    New-Item -ItemType Directory -Path $directory | Out-Null
    $vars = Join-Path $directory 'OVMF_VARS.fd'
    Copy-Item -LiteralPath $config.firmware_vars -Destination $vars
    Write-ReimsJson $manifest ([ordered]@{
        schema = 1; name = $Name; created_utc = [DateTime]::UtcNow.ToString('o')
        disk = $config.active_disk; firmware_vars = $vars
        firmware_code_sha256 = (Get-FileHash -LiteralPath $config.firmware_code).Hash
        opencore_sha256 = (Get-FileHash -LiteralPath $config.opencore).Hash
    })
} else {
    if (!(Test-Path -LiteralPath $manifest)) { throw "Snapshot not found: $Name" }
    $snapshot = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
    if ($snapshot.schema -ne 1) { throw 'Unknown snapshot schema.' }
    [void](Assert-ReimsArtifactPath $snapshot.disk)
    [void](Assert-ReimsArtifactPath $snapshot.firmware_vars)
    if ((Get-FileHash -LiteralPath $config.firmware_code).Hash -ne $snapshot.firmware_code_sha256 -or
        (Get-FileHash -LiteralPath $config.opencore).Hash -ne $snapshot.opencore_sha256) {
        throw 'Firmware/OpenCore changed since this snapshot. Restore with the original firmware pair.'
    }
    Invoke-ReimsTool $QemuImg @('check', '-q', $snapshot.disk) | Out-Null
    Invoke-ReimsTool $QemuImg @('create', '-f', 'qcow2', '-F', 'qcow2', '-b', $snapshot.disk, $next) | Out-Null
    $vars = Join-Path $VmRoot ("firmware/OVMF_VARS-" + [guid]::NewGuid().ToString() + '.fd')
    Copy-Item -LiteralPath $snapshot.firmware_vars -Destination $vars
    $config.firmware_vars = $vars
}
$config.active_disk = $next
Write-ReimsJson (Join-Path $VmRoot 'state/config.json') $config
"$Action snapshot '$Name'; next boot writes $next. Previous disks are retained."

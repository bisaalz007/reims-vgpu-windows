#Requires -Version 7.2
. "$PSScriptRoot/../start.ps1" -FunctionsOnly
$testRoot=Join-Path $ProjectRoot ('.work/tmp/usb-test-'+[guid]::NewGuid())
$VmRoot=$testRoot
New-Item -ItemType Directory -Path $testRoot | Out-Null
$script:checks=0
function Assert($Condition, $Message) { if (!$Condition) { throw $Message }; $script:checks++ }
function Reject([scriptblock]$Action) { $failed=$false; try { & $Action | Out-Null } catch { $failed=$true }; Assert $failed 'Expected USB refusal' }
$script:inventory=@'
  Bus 2, Addr 12, Port 11.2, Speed 12 Mb/s
    Class 00: USB device 0fd9:0080
  Bus 2, Addr 18, Port 17, Speed 5000 Mb/s
    Class 00: USB device 0bda:9210
  Bus 6, Addr 2, Port 1, Speed 12 Mb/s
    Class 00: USB device 046d:c539
'@
$script:objects=@{}
$script:running=$true
function Get-ReimsVmConfig { Get-Content -Raw (Join-Path $VmRoot 'state/config.json') | ConvertFrom-Json }
function Get-ReimsRuntime { @{pid=1} }
function Get-ReimsVmProcess($Runtime) { $script:running }
function Get-PnpDevice { }
function Get-ReimsProtectedUsbIds {
    $blocked=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [void]$blocked.Add('0BDA:9210'); [void]$blocked.Add('046D:C539'); return ,$blocked
}
function Invoke-ReimsQmp($Runtime,$Command,$Arguments) {
    switch ($Command) {
        human-monitor-command { $script:inventory }
        device_add { $script:objects[$Arguments.id]=$Arguments }
        device_del { $script:objects.Remove($Arguments.id) }
        qom-list { foreach ($key in $script:objects.Keys) { [pscustomobject]@{name=$key} } }
        default { throw "Unexpected QMP: $Command" }
    }
}
try {
    Write-ReimsJson (Join-Path $VmRoot 'state/config.json') @{schema=1}
    $rows=@(Get-ReimsUsbDevices)
    Assert ($rows.Count -eq 1 -and $rows[0].port -eq '11.2') 'Host storage and input must be excluded'
    $id=$rows[0].id
    Set-ReimsUsbSelection $id $true | Out-Null
    $selected=@(Get-ReimsUsbSelections (Get-ReimsVmConfig))
    Assert ($selected.Count -eq 1 -and $script:objects.Count -eq 1) 'Only the selection is saved and attached'
    $argument=Get-ReimsUsbArgument $selected[0]
    Assert (!$argument.ContainsKey('hostaddr') -and $argument.hostport -eq '11.2' -and $argument.vendorid -eq 0xfd9 -and $argument.productid -eq 0x80) 'Reconnect must not use the transient address'
    Set-ReimsUsbSelection $id $true | Out-Null
    Assert (@(Get-ReimsUsbSelections (Get-ReimsVmConfig)).Count -eq 1) 'Repeated selection is idempotent'
    $script:inventory=$script:inventory.Replace('Addr 12','Addr 45')
    $rows=@(Get-ReimsUsbDevices)
    Assert ($rows[0].id -eq $id -and $rows[0].selected -and $rows[0].address -eq 45) 'Replug keeps selection despite changed address'
    $script:inventory=''
    $rows=@(Get-ReimsUsbDevices)
    Assert ($rows.Count -eq 1 -and !$rows[0].present -and $rows[0].selected) 'Unplugged selection remains removable'
    Set-ReimsUsbSelection $id $false | Out-Null
    Assert (@(Get-ReimsUsbSelections (Get-ReimsVmConfig)).Count -eq 0 -and !$script:objects.Count) 'Detach removes persisted and live watcher'
    $bad=[pscustomobject]@{id='hostusb-2-17-0bda9210';bus=2;port='17';vendor='0bda';product='9210';label='host storage'}
    Reject { Assert-ReimsUsbSafe @($bad) }
    $bad.port='17,hostaddr=18'
    Reject { Get-ReimsUsbSelections ([pscustomobject]@{usb_devices=@($bad)}) }
    $bad.port='17'; $bad.vendor='0000'
    Reject { Get-ReimsUsbSelections ([pscustomobject]@{usb_devices=@($bad)}) }
    $bad.vendor='0bda'
    Reject { Get-ReimsUsbSelections ([pscustomobject]@{usb_devices=@($bad,$bad)}) }
    $script:running=$false
    Write-ReimsJson (Join-Path $VmRoot 'state/config.json') @{schema=1;usb_devices=$selected}
    Assert (@(Get-ReimsUsbDevices).Count -eq 1) 'Stopped VM lists saved devices'
    Set-ReimsUsbSelection $id $false | Out-Null
    Assert (@(Get-ReimsUsbSelections (Get-ReimsVmConfig)).Count -eq 0) 'Stopped VM permits unregistering absent devices'
    "$script:checks USB checks passed."
} finally {
    $resolved=[IO.Path]::GetFullPath($testRoot)
    $allowed=[IO.Path]::GetFullPath((Join-Path $ProjectRoot '.work/tmp'))+[IO.Path]::DirectorySeparatorChar
    if ($resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}

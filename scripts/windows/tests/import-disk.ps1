#Requires -Version 7.2
. "$PSScriptRoot/../start.ps1" -FunctionsOnly
$root=$ProjectRoot
$testRoot=Join-Path $root ('.work/tmp/import-test-'+[guid]::NewGuid())
$originalVmOverride=$env:REIMS_VM_ROOT_OVERRIDE
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $img=Get-ReimsTool 'qemu-img.exe'; $io=Get-ReimsTool 'qemu-io.exe'
    $source=Join-Path $testRoot 'source.vmdk'; $destination=Join-Path $testRoot 'Imported.data'
    Invoke-ReimsTool $img @('create','-f','vmdk',$source,'32M') | Out-Null
    Invoke-ReimsTool $io @('-f','vmdk','-c','write -P 0x7a 0 4096',$source) | Out-Null
    $hash=(Get-FileHash $source).Hash
    & "$PSScriptRoot/../manager-backend.ps1" -Root $root -VmDirectory $destination -Action ImportDisk -DiskPath $source | Out-Null
    $config=Get-Content -Raw (Join-Path $destination 'state/config.json') | ConvertFrom-Json
    foreach ($key in @('base_disk','active_disk','opencore','firmware_vars')) {
        if (!$config.$key.StartsWith($destination,[StringComparison]::OrdinalIgnoreCase)) { throw "Import escaped its VM: $key" }
    }
    if ((Get-FileHash $source).Hash -ne $hash) { throw 'Import modified the source VMDK.' }
    Invoke-ReimsTool $img @('compare','-f','vmdk','-F','qcow2',$source,$config.active_disk) | Out-Null
    Invoke-ReimsTool $io @('-f','qcow2','-c','write -P 0x5a 4096 4096',$config.active_disk) | Out-Null
    Invoke-ReimsTool $io @('-f','vmdk','-c','read -P 0 4096 4096',$source) | Out-Null
    $legacy=Get-Content -Raw (Join-Path $root 'vm/state/config.json') | ConvertFrom-Json
    if ($config.qmp_port -in @($legacy.qmp_port,$legacy.ssh_port) -or $config.ssh_port -in @($legacy.qmp_port,$legacy.ssh_port)) { throw 'Imported VM reused legacy ports.' }
    'VMDK import, source preservation, independent overlay and VM ports passed.'
} finally {
    $env:REIMS_VM_ROOT_OVERRIDE=$originalVmOverride
    $resolved=[IO.Path]::GetFullPath($testRoot)
    $allowed=[IO.Path]::GetFullPath((Join-Path $root '.work/tmp'))+[IO.Path]::DirectorySeparatorChar
    if ($resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) {
        try { Remove-Item -LiteralPath $resolved -Recurse -Force }
        catch { Write-Warning "Fixture retained because its generated SSH key is protected: $resolved" }
    }
}

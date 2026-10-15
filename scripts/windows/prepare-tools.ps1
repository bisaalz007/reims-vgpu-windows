#Requires -Version 7.2
. "$PSScriptRoot/start.ps1" -FunctionsOnly
Assert-ReimsVmStopped
$directory = Join-Path $VmRoot 'tools/root'
$key = Join-Path $VmRoot 'state/ssh/id_ed25519'
New-Item -ItemType Directory -Force -Path $directory, (Split-Path $key) | Out-Null
if (!(Test-Path $key)) {
    Invoke-ReimsTool "$env:SystemRoot/System32/OpenSSH/ssh-keygen.exe" @('-q','-t','ed25519','-N','','-C','reims-vm-control','-f',$key)
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    Invoke-ReimsTool "$env:SystemRoot/System32/icacls.exe" @($key,'/inheritance:r','/grant:r',"*$($sid):(R)") | Out-Null
}
Copy-Item -LiteralPath "$key.pub" -Destination (Join-Path $directory 'authorized_key')
Copy-Item -LiteralPath "$PSScriptRoot/guest-setup.command" -Destination (Join-Path $directory 'setup.command')
$image = Join-Path $VmRoot 'tools/ReimsTools.img'
Invoke-ReimsTool (Get-ReimsTool 'qemu-img.exe') @('convert','-f','vvfat','-O','raw',('fat:'+$directory.Replace('\','/')),"$image.partial") | Out-Null
$python = Get-ReimsTool 'python.exe'
@'
import pathlib, struct, sys
p=pathlib.Path(sys.argv[1])
with p.open('r+b') as f:
    mbr=f.read(512)
    start=struct.unpack_from('<I', mbr, 454)[0]*512
    f.seek(start); bpb=f.read(512)
    assert bpb[54:62].startswith(b'FAT'), 'Expected FAT12/16 tools volume'
    f.seek(start+43); f.write(b'REIMSTOOLS ')
    reserved=struct.unpack_from('<H',bpb,14)[0]
    fat_sectors=struct.unpack_from('<H',bpb,22)[0]
    entries=struct.unpack_from('<H',bpb,17)[0]
    root=start+(reserved+bpb[16]*fat_sectors)*512
    for index in range(entries):
        offset=root+index*32
        f.seek(offset); entry=f.read(32)
        if entry[11]&8 and entry[11]!=15:
            f.seek(offset); f.write(b'REIMSTOOLS ')
            break
    else:
        raise AssertionError('FAT root volume label missing')
'@ | & $python - "$image.partial"
if ($LASTEXITCODE -ne 0) { throw 'Tools volume label failed.' }
Move-Item -LiteralPath "$image.partial" -Destination $image -Force
'ReimsTools 準備完了。ゲストの Finder から setup.command を開いてください。'

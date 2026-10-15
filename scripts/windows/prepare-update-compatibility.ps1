#Requires -Version 7.2
[CmdletBinding()]
param()
. "$PSScriptRoot/start.ps1" -FunctionsOnly
Assert-ReimsVmStopped
$manifest = Get-Content -Raw (Join-Path $VmRoot 'state/opencore.json') | ConvertFrom-Json
[void](Assert-ReimsArtifactPath $manifest.config)
$ocDirectory = Split-Path $manifest.config
$packages = @(
    @{name='Lilu';version='1.7.2';sha256='53967d7dcfaab01023a33df2e969a89522f13d6654a6a56ac4711b62dabf3ab8'},
    @{name='RestrictEvents';version='1.1.6';sha256='98170dfae195ddd28b5d95e3f040125a13ca783bcb9bd1e5b8c588e217b14ee6'}
)
foreach ($package in $packages) {
    $archive = Join-Path $ProjectRoot ".work/downloads/$($package.name)-$($package.version)-RELEASE.zip"
    if (!(Test-Path -LiteralPath $archive)) {
        $url = "https://github.com/acidanthera/$($package.name)/releases/download/$($package.version)/$($package.name)-$($package.version)-RELEASE.zip"
        Invoke-WebRequest $url -OutFile "$archive.partial"
        if ((Get-FileHash "$archive.partial").Hash -ne $package.sha256) { throw "Download digest mismatch: $($package.name)" }
        Move-Item -LiteralPath "$archive.partial" -Destination $archive
    }
    if ((Get-FileHash $archive).Hash -ne $package.sha256) { throw "Cached digest mismatch: $($package.name)" }
    $expanded = Join-Path $ProjectRoot ".work/downloads/$($package.name)-$($package.version)"
    Expand-Archive -LiteralPath $archive -DestinationPath $expanded -Force
    $kext = Get-ChildItem -LiteralPath $expanded -Directory -Recurse -Filter "$($package.name).kext" | Select-Object -First 1
    if (!$kext) { throw "Missing kext: $($package.name)" }
    $destination = Join-Path $ocDirectory 'Kexts'
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Copy-Item -LiteralPath $kext.FullName -Destination $destination -Recurse -Force
}
$backup = Join-Path $ProjectRoot ('.work/diagnostics/opencore-before-updates-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff') + '.plist')
Copy-Item -LiteralPath $manifest.config -Destination $backup
$candidate = "$backup.candidate"
$patch = @'
import pathlib, plistlib, sys
source, destination = map(pathlib.Path,sys.argv[1:])
c=plistlib.loads(source.read_bytes())
names=['Lilu','RestrictEvents']
entries=[dict(Arch='x86_64',BundlePath=n+'.kext',Comment=n+' update compatibility',Enabled=True,
    ExecutablePath='Contents/MacOS/'+n,MinKernel='25.0.0',MaxKernel='25.99.99',PlistPath='Contents/Info.plist') for n in names]
c['Kernel']['Add']=entries+[p for p in c['Kernel']['Add'] if p['BundlePath'] not in [n+'.kext' for n in names]]
c['Kernel']['Quirks']['DisableLinkeditJettison']=True
nv=c['NVRAM']['Add']['7C436110-AB2A-4BBB-A880-FE41995C9F82']
args=[a for a in nv.get('boot-args','').split() if not a.startswith(('revpatch=','revblock='))]
nv['boot-args']=' '.join(args+['revpatch=sbvmm','revblock=none'])
destination.write_bytes(plistlib.dumps(c,sort_keys=False))
'@
$patch | & (Get-ReimsTool 'python.exe') - $manifest.config $candidate
if ($LASTEXITCODE -ne 0) { throw 'Update compatibility configuration failed.' }
Invoke-ReimsTool $manifest.ocvalidate @($candidate) | Out-Null
Move-Item -LiteralPath $candidate -Destination $manifest.config -Force
& "$PSScriptRoot/prepare-opencore.ps1" -Rebuild | Out-Null
"更新検出の互換設定を保存しました。次回起動後に macOS の更新を確認してください。バックアップ: $backup"

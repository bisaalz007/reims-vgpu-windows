#Requires -Version 7.2
[CmdletBinding()]
param()
. "$PSScriptRoot/start.ps1" -FunctionsOnly
Assert-ReimsVmStopped
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'USB 転送ドライバーの導入には Windows の管理者権限が必要です。' }
$installer = Join-Path $ProjectRoot '.work/downloads/UsbDk_1.0.22_x64.msi'
if (!(Test-Path -LiteralPath $installer)) {
    Invoke-WebRequest 'https://github.com/daynix/UsbDk/releases/download/v1.00-22/UsbDk_1.0.22_x64.msi' -OutFile $installer
}
if ((Get-FileHash $installer).Hash -ne '91F6F695E1E13C656024E6D3B55620BF08D8835EF05EE0496935BA6BB62466A5') { throw 'UsbDk のダウンロードが検証済みファイルと一致しません。' }
$signature = Get-AuthenticodeSignature $installer
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Red Hat, Inc\.') { throw 'UsbDk の署名を確認できません。' }
$log = Join-Path $ProjectRoot '.work/diagnostics/usbdk-install.log'
$process = Start-Process -FilePath "$env:SystemRoot/System32/msiexec.exe" -ArgumentList "/i `"$installer`" /qn /norestart /L*v `"$log`"" -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -notin @(0,3010)) { throw "USB 転送ドライバーの導入に失敗しました ($($process.ExitCode))。ログ: $log" }
if ($process.ExitCode -eq 3010) { '導入しました。USB 転送を使用する前に Windows を再起動してください。' }
else { 'USB 転送ドライバーを導入しました。次回 VM 起動時に使用します。' }

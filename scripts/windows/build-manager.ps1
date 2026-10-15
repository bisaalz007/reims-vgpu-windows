#Requires -Version 7.2
param([switch]$SelfTest, [string]$Dotnet)
. "$PSScriptRoot/environment.ps1"
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../apps/windows-manager/ReimsManager.csproj'))
$directory = Join-Path $ProjectRoot 'build/windows-manager'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$exe = Join-Path $directory 'ReimsManager.exe'
if (!$Dotnet) { $Dotnet = Join-Path $ProjectRoot '.work/dotnet-sdk/dotnet.exe' }
if (!(Test-Path $Dotnet)) { throw 'Pass -Dotnet with a Windows .NET 10 SDK. The published app needs no separate runtime.' }
$env:DOTNET_CLI_HOME = Join-Path $ProjectRoot '.work/dotnet-home'
$env:NUGET_PACKAGES = Join-Path $ProjectRoot '.work/cache/nuget'
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$intermediate = Join-Path $ProjectRoot 'build/windows-manager-obj/'
$compiled = Join-Path $ProjectRoot 'build/windows-manager-compiled/'
& $Dotnet publish $project -c Release -r win-x64 --self-contained true -o $directory "-p:BaseIntermediateOutputPath=$intermediate" "-p:BaseOutputPath=$compiled"
if ($LASTEXITCODE -ne 0) { throw 'Fluent manager build failed.' }
if ($SelfTest) {
    $test = Start-Process -FilePath $exe -ArgumentList '--self-test', ('"' + (Join-Path $ProjectRoot '.work/diagnostics/manager-self-test') + '"') -WindowStyle Hidden -PassThru -Wait
    if ($test.ExitCode -ne 0) { throw 'Manager native self-test failed.' }
}
"Built: $exe"

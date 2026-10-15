#Requires -Version 7.2
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Command,
    [hashtable]$Arguments = @{}
)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
$runtime = Get-ReimsRuntime
if (!$runtime) { throw 'No recorded VM. Run start.ps1 first.' }
Invoke-ReimsQmp $runtime $Command $Arguments | ConvertTo-Json -Depth 20

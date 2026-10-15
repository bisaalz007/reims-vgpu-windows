$systemTemp = [Environment]::GetEnvironmentVariable('TEMP', 'User')
. (Join-Path $PSScriptRoot '..\environment.ps1')
foreach ($name in @('TEMP', 'TMP', 'TMPDIR', 'CARGO_HOME', 'RUSTUP_HOME', 'CARGO_TARGET_DIR', 'XDG_CACHE_HOME', 'PIP_CACHE_DIR', 'CCACHE_DIR')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    if (-not $value.StartsWith($ProjectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "$name escapes the project: $value"
    }
}
if ([Environment]::GetEnvironmentVariable('TEMP', 'User') -ne $systemTemp) { throw 'User TEMP changed' }
Write-Output 'PASS: project paths and process-only environment'

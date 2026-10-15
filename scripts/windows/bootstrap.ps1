. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $ProjectRoot
try {
    git config core.autocrlf false
    if ($LASTEXITCODE -ne 0) { throw 'Cannot configure repository line endings' }
    git submodule update --init vendor/qemu
    if ($LASTEXITCODE -ne 0) { throw 'QEMU source checkout failed' }
    $packages = @(
        'make', 'diffutils', 'bison', 'flex',
        'clang', 'llvm', 'llvm-tools', 'lld', 'cmake', 'meson', 'ninja', 'python',
        'glib2', 'pixman', 'pkgconf', 'libslirp', 'libusb', 'SDL2', 'vulkan-headers',
        'vulkan-loader', 'vulkan-validation-layers', 'spirv-tools'
    ) | ForEach-Object { if ($_ -in @('make', 'diffutils', 'bison', 'flex')) { $_ } else { 'mingw-w64-ucrt-x86_64-' + $_ } }
    Invoke-ReimsMsys ('mkdir -p "$XDG_CACHE_HOME/pacman" && pacman -S --noconfirm --needed --cachedir "$XDG_CACHE_HOME/pacman" --logfile "$TEMP/../diagnostics/pacman-install.log" ' + ($packages -join ' '))
    rustup toolchain install $env:RUSTUP_TOOLCHAIN --profile minimal --component rustfmt --component clippy
    if ($LASTEXITCODE -ne 0) { throw 'Pinned native Rust install failed' }
    cargo install --locked cbindgen --version 0.29.4
    if ($LASTEXITCODE -ne 0) { throw 'cbindgen install failed' }
    Invoke-ReimsMsys 'pacman -Q > "$TEMP/../diagnostics/msys2-packages-current.txt"'
    & (Join-Path $PSScriptRoot 'doctor.ps1')
} finally { Pop-Location }

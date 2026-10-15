param([ValidateSet('dev', 'release', 'checked')][string]$Profile = 'release')
. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $ProjectRoot
try {
    Invoke-ReimsMsys 'MSYS2_ARG_CONV_EXCL="*" make -C firmware OS=Windows_NT BUILD="$(cygpath -m "$CARGO_TARGET_DIR")/firmware" rom'
    cargo xtask dist --platform vulkan --profile $Profile
    if ($LASTEXITCODE -ne 0) { throw 'Native Reims distribution build failed' }
    $env:PKG_CONFIG_PATH = Join-Path $env:CARGO_TARGET_DIR 'dist-vulkan\lib\pkgconfig'
    New-Item -ItemType Directory -Path (Join-Path $ProjectRoot 'build\qemu') -Force | Out-Null
    Invoke-ReimsMsys 'cd build/qemu && PKG_CONFIG_PATH="$(cygpath -u "$CARGO_TARGET_DIR")/dist-vulkan/lib/pkgconfig:$PKG_CONFIG_PATH" WINDRES=llvm-windres CC_LD=lld CXX_LD=lld ../../vendor/qemu/configure --target-list=x86_64-softmmu --cc=clang --cxx=clang++ --extra-ldflags=-fuse-ld=lld --enable-whpx --disable-kvm --disable-hvf --enable-pvg --enable-libusb --enable-sdl --enable-slirp --disable-gtk --disable-vnc --disable-spice --disable-gnutls --disable-curses --disable-nettle --disable-rust --disable-werror && ninja -j4 qemu-system-x86_64.exe qemu-img.exe qemu-io.exe'
    Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR 'dist-vulkan\lib\reims_vgpu_c.dll') -Destination (Join-Path $ProjectRoot 'build\qemu')
    Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR 'dist-vulkan\share\reims-vgpu\ReimsVgpu.rom') -Destination (Join-Path $ProjectRoot 'build\qemu')
    $manifest = [ordered]@{
        Qemu = 'build/qemu/qemu-system-x86_64.exe'
        QemuImg = 'build/qemu/qemu-img.exe'
        Dist = '.work/target/dist-vulkan'
        Profile = $Profile
        ReimsCommit = (git rev-parse HEAD)
        QemuCommit = (git -C vendor/qemu rev-parse HEAD)
    }
    $manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ProjectRoot '.work\diagnostics\build-manifest.json') -Encoding UTF8
} finally { Pop-Location }

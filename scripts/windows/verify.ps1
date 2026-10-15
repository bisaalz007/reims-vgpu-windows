param([switch]$RequireWhpx)
. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $ProjectRoot
try {
    & (Join-Path $PSScriptRoot 'tests\environment.ps1')
    $probe = Join-Path $ProjectRoot '.work\diagnostics\windows-vm-probe'
    New-Item -ItemType Directory -Path (Join-Path $probe 'src') -Force | Out-Null
    $vulkanPath = (Join-Path $ProjectRoot 'crates\reims-vgpu-vulkan').Replace('\', '/')
    $corePath = (Join-Path $ProjectRoot 'crates\reims-vgpu-core').Replace('\', '/')
    @"
[package]
name = "windows-vm-probe"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
reims-vgpu-vulkan = { path = "$vulkanPath" }
reims-vgpu-core = { path = "$corePath" }
"@ | Set-Content -LiteralPath (Join-Path $probe 'Cargo.toml') -Encoding UTF8
    @'
use reims_vgpu_vulkan::os::OsVm;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::os::vm::{HostVm, VmProt};
fn main() {
    let vm = OsVm::new();
    let page = vm.page_size();
    assert_eq!(page, 4096);
    let source = vm.allocate_anywhere(3 * page).unwrap();
    let destination = vm.allocate_anywhere(3 * page).unwrap();
    let middle = HostPtr::from_addr(source.addr() + page as usize);
    unsafe { middle.as_ptr::<u8>().write(7); }
    vm.remap_shared_overwrite(destination, middle, page).unwrap();
    unsafe {
        assert_eq!(destination.as_ptr::<u8>().read(), 7);
        destination.as_ptr::<u8>().write(9);
        assert_eq!(middle.as_ptr::<u8>().read(), 9);
    }
    assert!(vm.deallocate(source, 3 * page).is_success());
    unsafe { assert_eq!(destination.as_ptr::<u8>().read(), 9); }
    assert!(vm.remap_shared_overwrite(destination, HostPtr::NULL, page).is_err());
    unsafe { assert_eq!(destination.as_ptr::<u8>().read(), 9); }
    let middle = HostPtr::from_addr(destination.addr() + page as usize);
    unsafe { middle.as_ptr::<u8>().write(5); }
    vm.allocate_fixed_overwrite(middle, page).unwrap();
    unsafe {
        assert_eq!(middle.as_ptr::<u8>().read(), 0);
        assert_eq!(destination.as_ptr::<u8>().read(), 9);
    }
    vm.protect_max(middle, page, VmProt::READ).unwrap();
    assert!(vm.protect_max(middle, page, VmProt::READ | VmProt::WRITE).is_err());
    assert!(vm.deallocate(destination, 3 * page).is_success());
    assert!(!vm.readable(destination, 1));
    println!("PASS: native shared aliases, partial overwrite, lifetime and protections");
}
'@ | Set-Content -LiteralPath (Join-Path $probe 'src\main.rs') -Encoding UTF8
    cargo run --manifest-path (Join-Path $probe 'Cargo.toml')
    if ($LASTEXITCODE -ne 0) { throw 'Native Windows memory verification failed' }
    & (Join-Path $PSScriptRoot 'doctor.ps1') -RequireWhpx:$RequireWhpx
} finally { Pop-Location }

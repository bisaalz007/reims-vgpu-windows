//! Windows distribution: the native Vulkan DLL and its GNU import library for QEMU.

use super::{LIB, Options, Platform, ROM, c_api_args, install, layout, pc_file, strs, write};
use crate::tools::Tools;
use crate::util::Res;
use std::path::{Path, PathBuf};

pub const DEFAULT_PLATFORM: Platform = Platform::Vulkan;
pub const SHARED_LIBRARY_EXTENSION: &str = "dll";

fn windows_pc(version: &str) -> String {
    // The import library retains Cargo's DLL name, so both keep that name in the dist tree.
    pc_file(version, &[]).replace(&format!("-l{LIB}\n"), &format!("-l{LIB}_c.dll\n"))
}

pub fn dist(
    tools: &mut dyn Tools,
    options: &Options,
    target: &Path,
    header: &Path,
    firmware: &Path,
) -> Res<PathBuf> {
    if options.platform != Platform::Vulkan {
        return Err("Windows builds the Vulkan platform only".into());
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut args = strs(&["build"]);
    args.extend(c_api_args(options));
    tools.run(&cargo, &args)?;
    let built = target.join(options.profile.dir());
    let [archive, shared, pc, include, rom] = layout().map(|p| options.out.join(p));
    install(&built.join("libreims_vgpu_c.a"), &archive, 0o644)?;
    install(&built.join("reims_vgpu_c.dll"), &shared, 0o755)?;
    install(
        &built.join("libreims_vgpu_c.dll.a"),
        &options.out.join("lib/libreims_vgpu_c.dll.a"),
        0o644,
    )?;
    install(header, &include, 0o644)?;
    write(&pc, &windows_pc(env!("CARGO_PKG_VERSION")))?;
    let _ = firmware;
    install(&target.join("firmware").join(ROM), &rom, 0o644).map_err(|e| {
        format!("{e}: build the Windows option ROM with scripts/windows/build.ps1 first")
    })?;
    Ok(options.out.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn package_links_the_dll_import_library() {
        let package = windows_pc("1.0");
        assert!(package.contains("Libs: -L${libdir} -lreims_vgpu_c.dll\n"));
        assert_eq!(layout()[1], "lib/reims_vgpu_c.dll");
    }
}

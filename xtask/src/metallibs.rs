//! `cargo xtask metallibs [<dir>]`: the Vulkan platform's metallibs, compiled on a Mac into a
//! directory a build elsewhere takes them from. The runtime makes libraries of metallibs, and only
//! Apple's tools compile Metal's language into them, so a build on Linux copies these instead,
//! once `reims-vgpu-vulkan/build.rs` has checked the directory's `metallibs.toml` against the
//! tree's sources. With no `<dir>` it writes the directory the repository holds,
//! `crates/reims-vgpu-vulkan/metallibs` ([`COMMITTED`]), which a build elsewhere takes by default;
//! a `<dir>` of its own is for `RVG_METALLIB_DIR`.
//!
//! The command runs that build script with `RVG_METALLIB_EXPORT` set to `<dir>`: it compiles as
//! it does on every Mac build, then copies each portable metallib and the record of what it was
//! built from into `<dir>`.

use crate::util::{Res, repo_root};

/// The usage line.
pub const USAGE: &str = "usage: cargo xtask metallibs [<dir>]";

/// The directory of metallibs the repository holds, from its root.
pub const COMMITTED: &str = "crates/reims-vgpu-vulkan/metallibs";

/// `cargo xtask metallibs`.
pub fn cmd_metallibs(args: &[String]) -> Res<bool> {
    let dir = match args {
        [] => repo_root().join(COMMITTED),
        [dir] => std::path::absolute(dir).map_err(|e| format!("{dir}: {e}"))?,
        _ => return Err(USAGE.to_owned()),
    };
    if !cfg!(target_os = "macos") {
        return Err("the metallibs are compiled with Xcode's Metal toolchain, on a Mac".to_owned());
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    println!(
        "$ RVG_METALLIB_EXPORT={} {cargo} check -p reims-vgpu-vulkan",
        dir.display()
    );
    let status = std::process::Command::new(cargo)
        .args(["check", "-p", "reims-vgpu-vulkan"])
        .env("RVG_METALLIB_EXPORT", &dir)
        .env_remove("RVG_METALLIB_DIR")
        .current_dir(repo_root())
        .status()
        .map_err(|e| format!("cargo: {e}"))?;
    if status.success() {
        println!("metallibs: {}", dir.join("metallibs.toml").display());
    }
    Ok(status.success())
}

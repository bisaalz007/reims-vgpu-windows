//! Paths, file IO and name helpers shared by every command.

use std::path::{Path, PathBuf};

/// Every fallible step reports a message; `main` prints it and fails.
pub type Res<T> = Result<T, String>;

/// The repository root, which is also the workspace root: the parent of `xtask/`.
pub fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

pub fn read(path: &Path) -> Res<String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes `content` unless the file already holds it. Returns whether it wrote.
pub fn write_if_changed(path: &Path, content: &str) -> Res<bool> {
    if let Ok(old) = std::fs::read_to_string(path)
        && old == content
    {
        return Ok(false);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, content).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(true)
}

/// Every file under `dir` (recursively) whose name ends with one of `exts`, sorted, skipping
/// `target` directories.
pub fn walk(dir: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    stack.push(p);
                }
            } else if exts.iter().any(|x| name.ends_with(x)) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// `path` relative to `base`, with `/` separators, or the whole path when it is not under it.
pub fn rel(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Runs a command and returns its stdout, failing on a non-zero exit.
pub fn run_capture(cmd: &mut std::process::Command) -> Res<String> {
    let out = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{cmd:?} failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

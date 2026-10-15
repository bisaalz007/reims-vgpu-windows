//! `cargo xtask smoke` (`make smoke`): a guest workload run against the library, linked the way
//! `make qemu` links QEMU.
//!
//! One program runs, with the caller's `DYLD_*` variables dropped and for at most [`TIMEOUT`]:
//! **the guest** (`smoke_guest.m`), compiled against the dist tree's `reims_vgpu.h` and linked
//! with its `libreims_vgpu.a` and the `Libs.private` of its `reims-vgpu.pc`, so the library is
//! part of the guest's own image. It makes a device over a memory map with process isolation, as
//! QEMU's apple-gfx-mmio makes it, and drives it through MMIO and the root FIFO: bring-up, a NOP,
//! then task 1 defined, mapped, given an object list, and one exec whose fill must land in guest
//! RAM. Every fact must equal the library's own expected values, [`GUEST_EXPECTED`], and
//! `rvg_device_create` must be the guest's own.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::dist::{HEADER, LIB, PACKAGE};
use crate::tools::{MIN_MACOS, System, Tools, target_dir};
use crate::util::{Res, repo_root};

/// How long the guest may run.
pub const TIMEOUT: Duration = Duration::from_secs(120);

/// The usage line.
pub const USAGE: &str = "usage: cargo xtask smoke [--dist DIR] [--out DIR] [--timeout SECONDS]";

/// The guest program's facts that are behaviour, and the values the library must give.
pub const GUEST_EXPECTED: &[(&str, &str)] = &[
    ("device", "created"),
    ("version", "245"),
    ("root_page", "ok"),
    ("fifo_enabled", "1"),
    ("fault_offset", "0xffffffff"),
    ("nop", "ok"),
    ("read", "12"),
    ("interrupt", "raised"),
    ("task_define", "ok"),
    ("task_handshake", "accepted"),
    ("task_root_word", "ok"),
    ("task_length", "ok"),
    ("task_map", "ok"),
    ("task_objects", "ok"),
    ("task_exec", "ok"),
    ("task_fill", "ok"),
    ("faulted", "0"),
];

// ---------------------------------------------------------------------------------------------
// Options

/// `cargo xtask smoke`'s arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// The dist tree (`cargo xtask dist`).
    pub dist: PathBuf,
    /// Where the program and its output go.
    pub out: PathBuf,
    pub timeout: Duration,
}

/// Parses the arguments. The dist tree is taken from where `cargo xtask dist` puts it by default.
pub fn parse_args(args: &[String], target_dir: &Path) -> Res<Options> {
    let mut o = Options {
        dist: target_dir.join("dist"),
        out: target_dir.join("smoke"),
        timeout: TIMEOUT,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--dist" => o.dist = PathBuf::from(value("--dist")?),
            "--out" => o.out = PathBuf::from(value("--out")?),
            "--timeout" => {
                let v = value("--timeout")?;
                let secs: u64 = v
                    .parse()
                    .map_err(|_| format!("--timeout {v}: not a number of seconds\n{USAGE}"))?;
                o.timeout = Duration::from_secs(secs);
            }
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(o)
}

// ---------------------------------------------------------------------------------------------
// The dist tree

/// The dist tree the guest links.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dist {
    pub dir: PathBuf,
}

impl Dist {
    /// The directory holding `reims_vgpu.h`.
    pub fn include(&self) -> PathBuf {
        self.dir.join("include")
    }

    /// The static library.
    pub fn archive(&self) -> PathBuf {
        self.dir.join("lib").join(format!("lib{LIB}.a"))
    }

    /// The package file.
    pub fn pc(&self) -> PathBuf {
        self.dir.join("lib/pkgconfig").join(format!("{PACKAGE}.pc"))
    }
}

/// The words of a `.pc` file's `Libs.private` line, or `None` where it has none.
pub fn libs_private(pc: &str) -> Option<Vec<String>> {
    pc.lines()
        .find_map(|l| l.strip_prefix("Libs.private:"))
        .map(|libs| libs.split_whitespace().map(str::to_owned).collect())
}

/// `p` with symlinks resolved, or `p` itself when it does not exist.
pub fn real(p: &str) -> String {
    std::fs::canonicalize(p)
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_else(|_| p.to_owned())
}

/// `path` relative to `base` when inside it.
pub fn shown(path: &str, base: &Path) -> String {
    let root = format!("{}/", base.to_string_lossy());
    path.strip_prefix(&root).unwrap_or(path).to_owned()
}

// ---------------------------------------------------------------------------------------------
// The guest

/// `smoke_guest.m`.
pub fn guest_source() -> PathBuf {
    repo_root().join("xtask/src/smoke_guest.m")
}

/// The `xcrun` arguments that compile the guest into `exe` and link the library into it
/// statically, with `libs` (its `Libs.private`); arm64 only.
pub fn clang_args(d: &Dist, libs: &[String], exe: &Path) -> Vec<String> {
    let mut v: Vec<String> = vec![
        "clang".to_owned(),
        "-arch".to_owned(),
        "arm64".to_owned(),
        format!("-mmacosx-version-min={MIN_MACOS}"),
        "-I".to_owned(),
        d.include().to_string_lossy().into_owned(),
        guest_source().to_string_lossy().into_owned(),
        d.archive().to_string_lossy().into_owned(),
    ];
    v.extend(libs.iter().cloned());
    v.extend(["-framework".to_owned(), "Foundation".to_owned()]);
    v.push("-o".to_owned());
    v.push(exe.to_string_lossy().into_owned());
    v
}

/// What a run printed and how it ended.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ran {
    pub stdout: String,
    pub stderr: String,
    /// The exit status (`0`, `1`, `signal 6`), or `timed out after N s`.
    pub exit: String,
}

/// The external programs: [`Tools`] for `xcrun`, and a timed run of a built program.
pub trait Host: Tools {
    /// Runs `exe` with `args`, the caller's `DYLD_*` variables removed, killing it after
    /// `timeout`; its stdout and stderr go to `log` and `log.err`.
    fn exec(&mut self, exe: &Path, args: &[String], timeout: Duration, log: &Path) -> Res<Ran>;
}

impl Host for System {
    fn exec(&mut self, exe: &Path, args: &[String], timeout: Duration, log: &Path) -> Res<Ran> {
        let err_log = log.with_extension("txt.err");
        let open = |p: &Path| std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()));
        let mut cmd = std::process::Command::new(exe);
        cmd.args(args)
            .stdin(std::process::Stdio::null())
            .stdout(open(log)?)
            .stderr(open(&err_log)?);
        for k in dyld_vars(std::env::vars_os().map(|(k, _)| k)) {
            cmd.env_remove(k);
        }
        let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", exe.display()))?;
        let start = Instant::now();
        let exit = loop {
            match child
                .try_wait()
                .map_err(|e| format!("{}: {e}", exe.display()))?
            {
                Some(status) => break exit_text(status),
                None if start.elapsed() >= timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break format!("timed out after {} s", timeout.as_secs());
                }
                None => std::thread::sleep(Duration::from_millis(100)),
            }
        };
        let read = |p: &Path| std::fs::read(p).map(|b| String::from_utf8_lossy(&b).into_owned());
        Ok(Ran {
            stdout: read(log).unwrap_or_default(),
            stderr: read(&err_log).unwrap_or_default(),
            exit,
        })
    }
}

/// The `DYLD_*` names among `vars`: what a run drops from the caller's environment.
pub fn dyld_vars(vars: impl Iterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    vars.filter(|k| k.to_string_lossy().starts_with("DYLD_"))
        .collect()
}

fn exit_text(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        match (status.code(), status.signal()) {
            (Some(c), _) => c.to_string(),
            (None, Some(s)) => format!("signal {s}"),
            (None, None) => status.to_string(),
        }
    }
    #[cfg(windows)]
    {
        status
            .code()
            .map_or_else(|| status.to_string(), |c| c.to_string())
    }
}

// ---------------------------------------------------------------------------------------------
// Facts

/// A program's `key: value` lines and how it ended.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    pub lines: Vec<(String, String)>,
    pub exit: String,
    pub stderr: String,
}

impl Facts {
    pub fn parse(ran: &Ran) -> Facts {
        let lines = ran
            .stdout
            .lines()
            .filter_map(|l| {
                let (k, v) = l.split_once(": ")?;
                Some((k.to_owned(), v.to_owned()))
            })
            .collect();
        Facts {
            lines,
            exit: ran.exit.clone(),
            stderr: ran.stderr.clone(),
        }
    }

    /// The first value printed for `key`, or `(none)`.
    pub fn get(&self, key: &str) -> &str {
        self.lines
            .iter()
            .find(|(k, _)| k == key)
            .map_or("(none)", |(_, v)| v.as_str())
    }

    /// Whether it printed `done: ok`.
    pub fn completed(&self) -> bool {
        self.get("done") == "ok"
    }

    /// The problem of a run that did not complete: its exit and the tail of its stderr.
    pub fn incomplete(&self) -> String {
        let tail: String = {
            let chars: Vec<char> = self.stderr.chars().collect();
            chars
                .get(chars.len().saturating_sub(2000)..)
                .unwrap_or(&[])
                .iter()
                .collect()
        };
        format!("exited {}: {}", self.exit, tail.trim())
    }
}

/// The guest's problems: against [`GUEST_EXPECTED`], and whether `rvg_device_create` is part of
/// `exe`, the guest itself.
pub fn check_guest(ours: &Facts, exe: &Path) -> Vec<String> {
    if !ours.completed() {
        return vec![format!("the guest {}", ours.incomplete())];
    }
    let mut p = Vec::new();
    let got = ours.get("image_entry");
    if real(got) != real(&exe.to_string_lossy()) {
        p.push(format!("image_entry is {got}, not the guest"));
    }
    for (key, want) in GUEST_EXPECTED {
        let got = ours.get(key);
        if got != *want {
            let offset = ours.get(&format!("{key}_fault_offset"));
            let detail = if offset == "(none)" {
                String::new()
            } else {
                format!(" (fault offset {offset})")
            };
            p.push(format!("{key} is {got}, not {want}{detail}"));
        }
    }
    p
}

/// The report lines of the guest's run.
pub fn summary(d: &Dist, g: &Facts, repo: &Path) -> Vec<String> {
    vec![
        format!(
            "smoke (arm64): the guest links {} statically",
            shown(&d.archive().to_string_lossy(), repo)
        ),
        format!(
            "  guest task 1: handshake {}, exec {}, fill {}",
            g.get("task_handshake"),
            g.get("task_exec"),
            g.get("task_fill"),
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// The run

/// Runs the smoke; returns the report lines and the problems.
pub fn smoke(host: &mut dyn Host, o: &Options, repo: &Path) -> Res<(Vec<String>, Vec<String>)> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("the smoke is arm64 macOS only".to_owned());
    }
    let d = Dist {
        dir: o.dist.clone(),
    };
    let missing = [d.archive(), d.pc(), d.include().join(HEADER)]
        .into_iter()
        .find(|p| !p.is_file());
    if let Some(p) = missing {
        let p = format!("no {}: run `cargo xtask dist`", p.display());
        return Ok((Vec::new(), vec![p]));
    }
    let pc = std::fs::read_to_string(d.pc()).map_err(|e| format!("{}: {e}", d.pc().display()))?;
    let libs =
        libs_private(&pc).ok_or_else(|| format!("{}: no Libs.private line", d.pc().display()))?;
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    let exe = o.out.join("guest-arm64");
    host.run("xcrun", &clang_args(&d, &libs, &exe))?;
    let ran = host.exec(&exe, &[], o.timeout, &o.out.join("guest-arm64.txt"))?;
    let guest = Facts::parse(&ran);
    let problems = check_guest(&guest, &exe)
        .into_iter()
        .map(|p| format!("guest: {p}"))
        .collect();
    Ok((summary(&d, &guest, repo), problems))
}

/// Renders the report: the summaries, each problem, then the verdict.
pub fn render(report: &[String], problems: &[String]) -> String {
    let mut s = String::new();
    for l in report {
        let _ = writeln!(s, "{l}");
    }
    for p in problems {
        let _ = writeln!(s, "  SMOKE: {p}");
    }
    let _ = writeln!(
        s,
        "smoke: {}",
        if problems.is_empty() { "ok" } else { "FAILED" }
    );
    s
}

/// `cargo xtask smoke`.
pub fn cmd_smoke(args: &[String]) -> Res<bool> {
    let target = target_dir(&mut System)?;
    let o = parse_args(args, &target)?;
    let (report, problems) = smoke(&mut System, &o, &repo_root())?;
    print!("{}", render(&report, &problems));
    Ok(problems.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_owned()).collect()
    }

    fn tmp(name: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "xtask-smoke-{}-{}-{name}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        // The checks compare resolved paths (`/var` is `/private/var`).
        std::fs::canonicalize(&p).unwrap()
    }

    fn put(p: &Path, content: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    fn opts(target: &Path) -> Options {
        parse_args(&[], target).unwrap()
    }

    fn ran(stdout: &str) -> Ran {
        Ran {
            stdout: stdout.to_owned(),
            stderr: String::new(),
            exit: "0".to_owned(),
        }
    }

    /// What the guest at `exe` prints when everything works.
    fn guest_out(exe: &Path) -> String {
        let mut out = format!("arch: arm64\nimage_entry: {}\n", exe.display());
        for (k, v) in GUEST_EXPECTED {
            out.push_str(&format!("{k}: {v}\n"));
        }
        out.push_str("done: ok\n");
        out
    }

    fn with(f: &Facts, key: &str, value: &str) -> Facts {
        let mut f = f.clone();
        for (k, v) in &mut f.lines {
            if k == key {
                *v = value.to_owned();
            }
        }
        f
    }

    // -- options ------------------------------------------------------------------------------

    #[test]
    fn defaults_smoke_the_dist_tree_where_dist_puts_it() {
        let o = opts(Path::new("/t"));
        assert_eq!(o.dist, PathBuf::from("/t/dist"));
        assert_eq!(o.out, PathBuf::from("/t/smoke"));
        assert_eq!(o.timeout, TIMEOUT);
        assert_eq!(TIMEOUT, Duration::from_secs(120), "two minutes");
    }

    #[test]
    fn every_option_is_read() {
        let o = parse_args(
            &s(&["--dist", "/a", "--out", "/d", "--timeout", "7"]),
            Path::new("/t"),
        )
        .unwrap();
        assert_eq!(o.dist, PathBuf::from("/a"));
        assert_eq!(o.out, PathBuf::from("/d"));
        assert_eq!(o.timeout, Duration::from_secs(7));
    }

    #[test]
    fn bad_arguments_fail_with_the_usage() {
        let t = Path::new("/t");
        let e = parse_args(&s(&["--bogus"]), t).unwrap_err();
        assert!(e.contains("unknown argument `--bogus`") && e.contains(USAGE));
        let e = parse_args(&s(&["--out"]), t).unwrap_err();
        assert!(e.contains("--out needs a value"));
        let e = parse_args(&s(&["--timeout", "soon"]), t).unwrap_err();
        assert!(e.contains("not a number"));
        for gone in ["--frameworks", "--skin", "--no-control"] {
            assert!(
                parse_args(&s(&[gone]), t)
                    .unwrap_err()
                    .contains(&format!("unknown argument `{gone}`"))
            );
        }
    }

    // -- the dist tree ------------------------------------------------------------------------

    #[test]
    fn the_dist_tree_is_dists_layout() {
        let d = Dist {
            dir: PathBuf::from("/t/dist"),
        };
        assert_eq!(d.archive(), PathBuf::from("/t/dist/lib/libreims_vgpu.a"));
        assert_eq!(d.pc(), PathBuf::from("/t/dist/lib/pkgconfig/reims-vgpu.pc"));
        assert_eq!(d.include(), PathBuf::from("/t/dist/include"));
        for p in [d.archive(), d.pc(), d.include().join(HEADER)] {
            let rel = p
                .strip_prefix("/t/dist")
                .unwrap()
                .to_string_lossy()
                .into_owned();
            assert!(
                crate::dist::layout()
                    .iter()
                    .any(|p| Path::new(p) == Path::new(&rel)),
                "{rel}"
            );
        }
    }

    #[test]
    fn libs_private_reads_the_package_line() {
        assert_eq!(
            libs_private("Libs: -lx\nLibs.private: -framework Metal  -lobjc\n"),
            Some(s(&["-framework", "Metal", "-lobjc"]))
        );
        assert_eq!(libs_private("Libs: -lx\n"), None);
    }

    #[cfg(unix)]
    #[test]
    fn paths_resolve_and_show() {
        let d = tmp("paths");
        put(&d.join("b/f"), "x");
        std::os::unix::fs::symlink(d.join("b"), d.join("link")).unwrap();
        assert_eq!(
            real(&d.join("link/f").to_string_lossy()),
            d.join("b/f").to_string_lossy()
        );
        assert_eq!(real("/no/such"), "/no/such");
        assert_eq!(shown("/repo/build/x", Path::new("/repo")), "build/x");
        assert_eq!(shown("/elsewhere/x", Path::new("/repo")), "/elsewhere/x");
    }

    // -- the guest program ----------------------------------------------------------------------

    #[test]
    fn the_guest_links_the_archive_and_its_private_libraries() {
        let d = Dist {
            dir: PathBuf::from("/t/dist"),
        };
        let v = clang_args(
            &d,
            &s(&["-framework", "Metal", "-lobjc"]),
            Path::new("/o/g"),
        );
        let src = guest_source();
        assert_eq!(
            v,
            s(&[
                "clang",
                "-arch",
                "arm64",
                "-mmacosx-version-min=26.6",
                "-I",
                &Path::new("/t/dist").join("include").to_string_lossy(),
                &src.to_string_lossy(),
                &Path::new("/t/dist")
                    .join("lib")
                    .join("libreims_vgpu.a")
                    .to_string_lossy(),
                "-framework",
                "Metal",
                "-lobjc",
                "-framework",
                "Foundation",
                "-o",
                "/o/g",
            ])
        );
        assert!(
            !v.iter().any(|a| a.contains("rpath")),
            "nothing to find at run time"
        );
    }

    #[test]
    fn the_guest_source_is_there() {
        assert!(guest_source().is_file());
    }

    // -- facts --------------------------------------------------------------------------------

    #[test]
    fn facts_take_the_first_value() {
        let f = Facts::parse(&ran("a: 1\na: 2\nnoise\nb c: (never)\n"));
        assert_eq!(f.get("a"), "1");
        assert_eq!(f.get("z"), "(none)");
        assert_eq!(f.get("b c"), "(never)");
        assert!(!f.completed());
        assert!(Facts::parse(&ran("done: ok\n")).completed());
    }

    #[test]
    fn an_incomplete_run_reports_its_exit_and_stderr_tail() {
        let f = Facts::parse(&Ran {
            stdout: "arch: arm64\n".to_owned(),
            stderr: format!("{}the end\n", "x".repeat(3000)),
            exit: "signal 6".to_owned(),
        });
        let m = f.incomplete();
        assert!(m.starts_with("exited signal 6: "));
        assert!(m.ends_with("the end"));
        assert!(m.len() < 2100, "the last 2000 characters");
    }

    // -- the guest's checks ---------------------------------------------------------------------

    #[test]
    fn a_good_guest_has_no_problem() {
        let exe = tmp("guest-good").join("g");
        put(&exe, "x");
        let g = Facts::parse(&ran(&guest_out(&exe)));
        assert_eq!(check_guest(&g, &exe), Vec::<String>::new());
    }

    #[test]
    fn an_incomplete_guest_is_one_problem() {
        let f = Facts::parse(&Ran {
            stdout: "arch: arm64\n".to_owned(),
            stderr: "boom".to_owned(),
            exit: "signal 11".to_owned(),
        });
        assert_eq!(
            check_guest(&f, Path::new("/g")),
            ["the guest exited signal 11: boom"]
        );
    }

    #[test]
    fn the_library_must_be_linked_into_the_guest() {
        let exe = tmp("guest-image").join("g");
        put(&exe, "x");
        let g = Facts::parse(&ran(&guest_out(&exe)));
        let g = with(&g, "image_entry", "/elsewhere/libreims_vgpu.dylib");
        assert_eq!(
            check_guest(&g, &exe),
            ["image_entry is /elsewhere/libreims_vgpu.dylib, not the guest"]
        );
    }

    #[test]
    fn a_refused_handshake_names_its_fault_offset() {
        let exe = Path::new("/g");
        let mut g = with(&Facts::parse(&ran(&guest_out(exe))), "task_define", "fault");
        g.lines
            .push(("task_define_fault_offset".to_owned(), "0x0".to_owned()));
        let g = with(&g, "task_handshake", "refused");
        let p = check_guest(&g, exe);
        assert!(p.contains(&"task_define is fault, not ok (fault offset 0x0)".to_owned()));
        assert!(p.contains(&"task_handshake is refused, not accepted".to_owned()));
    }

    #[test]
    fn every_expected_fact_is_checked() {
        let exe = Path::new("/g");
        for (key, want) in GUEST_EXPECTED {
            let g = with(&Facts::parse(&ran(&guest_out(exe))), key, "other");
            assert_eq!(
                check_guest(&g, exe),
                [format!("{key} is other, not {want}")],
                "{key}"
            );
        }
    }

    // -- reporting ----------------------------------------------------------------------------

    #[test]
    fn the_summary_names_the_archive() {
        let d = Dist {
            dir: PathBuf::from("/repo/target/dist"),
        };
        let g = Facts::parse(&ran(&guest_out(Path::new("/g"))));
        assert_eq!(
            summary(&d, &g, Path::new("/repo")),
            [
                &format!(
                    "smoke (arm64): the guest links {} statically",
                    Path::new("target/dist")
                        .join("lib")
                        .join("libreims_vgpu.a")
                        .display()
                ),
                "  guest task 1: handshake accepted, exec ok, fill ok",
            ]
        );
    }

    #[test]
    fn render_ends_with_the_verdict() {
        assert_eq!(render(&s(&["a"]), &[]), "a\nsmoke: ok\n");
        assert_eq!(render(&[], &s(&["x"])), "  SMOKE: x\nsmoke: FAILED\n");
    }

    // -- the run --------------------------------------------------------------------------------

    /// The runs, which link and run an arm64 macOS guest.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod run {
        use super::*;

        /// A dist tree on disk: the archive, the header and a package file.
        fn lay_out(dir: &Path) -> Dist {
            let d = Dist {
                dir: dir.to_path_buf(),
            };
            put(&d.archive(), "archive");
            put(&d.include().join(HEADER), "header");
            put(
                &d.pc(),
                "Name: reims-vgpu\nLibs.private: -framework Metal -lobjc\nCflags: -I${includedir}\n",
            );
            d
        }

        /// Records every call; answers each run with `output`, the guest's stdout.
        struct Fake {
            calls: Vec<String>,
            output: String,
        }

        impl Tools for Fake {
            fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
                self.calls.push(format!("{program} {}", args.join(" ")));
                Ok(())
            }
            fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
                self.calls.push(format!("{program} {}", args.join(" ")));
                Ok(String::new())
            }
        }

        impl Host for Fake {
            fn exec(
                &mut self,
                exe: &Path,
                args: &[String],
                timeout: Duration,
                log: &Path,
            ) -> Res<Ran> {
                assert_eq!(log, exe.with_extension("txt").as_path());
                assert_eq!(timeout, TIMEOUT);
                self.calls
                    .push(format!("exec {} {}", exe.display(), args.join(" ")));
                Ok(ran(&self.output))
            }
        }

        #[test]
        fn the_run_links_and_runs_the_guest() {
            let d = tmp("run");
            let mut o = opts(&d);
            o.out = d.join("out");
            lay_out(&o.dist);
            let exe = o.out.join("guest-arm64");
            let mut h = Fake {
                calls: Vec::new(),
                output: guest_out(&exe),
            };
            let (report, problems) = smoke(&mut h, &o, &d).unwrap();
            assert_eq!(problems, Vec::<String>::new(), "{report:?}");
            assert!(o.out.is_dir());
            assert_eq!(h.calls.len(), 2, "{:?}", h.calls);
            assert!(h.calls[0].starts_with("xcrun clang "));
            assert!(h.calls[0].contains("libreims_vgpu.a -framework Metal -lobjc -framework"));
            assert_eq!(h.calls[1], format!("exec {} ", exe.display()));
            assert_eq!(
                report[0],
                "smoke (arm64): the guest links dist/lib/libreims_vgpu.a statically"
            );
        }

        #[test]
        fn a_missing_dist_tree_is_a_problem() {
            let d = tmp("run-missing");
            let mut o = opts(&d);
            o.out = d.join("out");
            let mut h = Fake {
                calls: Vec::new(),
                output: String::new(),
            };
            let (_, problems) = smoke(&mut h, &o, &d).unwrap();
            assert_eq!(problems.len(), 1, "{problems:?}");
            assert!(problems[0].starts_with("no ") && problems[0].contains("libreims_vgpu.a"));
            assert!(problems[0].ends_with("run `cargo xtask dist`"));
            assert!(h.calls.is_empty(), "nothing is compiled without the tree");
        }

        #[test]
        fn a_failing_guest_is_reported() {
            let d = tmp("run-fail");
            let mut o = opts(&d);
            o.out = d.join("out");
            lay_out(&o.dist);
            let exe = o.out.join("guest-arm64");
            let bad =
                guest_out(&exe).replace("task_handshake: accepted", "task_handshake: refused");
            let mut h = Fake {
                calls: Vec::new(),
                output: bad,
            };
            let (_, problems) = smoke(&mut h, &o, &d).unwrap();
            assert_eq!(problems, ["guest: task_handshake is refused, not accepted"]);
        }
    }

    // -- the real host --------------------------------------------------------------------------

    #[cfg(unix)]
    fn sh(script: &str, timeout: Duration) -> Ran {
        let d = tmp("exec");
        System
            .exec(
                Path::new("/bin/sh"),
                &s(&["-c", script]),
                timeout,
                &d.join("x.txt"),
            )
            .unwrap()
    }

    #[test]
    fn exec_captures_output_and_exit() {
        #[cfg(unix)]
        {
            let r = sh("echo \"a: 1\"; echo err >&2; exit 3", TIMEOUT);
            assert_eq!(r.stdout, "a: 1\n");
            assert_eq!(r.stderr, "err\n");
            assert_eq!(r.exit, "3");
        }
        #[cfg(windows)]
        {
            let d = tmp("exec-cmd");
            let command = std::env::var_os("COMSPEC").unwrap();
            let r = System
                .exec(
                    Path::new(&command),
                    &s(&["/d", "/c", "echo a: 1& echo err 1>&2& exit /b 3"]),
                    TIMEOUT,
                    &d.join("x.txt"),
                )
                .unwrap();
            assert_eq!(r.stdout.trim(), "a: 1");
            assert_eq!(r.stderr.trim(), "err");
            assert_eq!(r.exit, "3");
        }
    }

    #[test]
    fn the_callers_dyld_variables_are_dropped() {
        // SIP strips DYLD_ variables from every system binary, so a child cannot show them; the
        // selection is checked here.
        let vars = [
            "PATH",
            "DYLD_FRAMEWORK_PATH",
            "XDYLD_A",
            "DYLD_INSERT_LIBRARIES",
        ]
        .iter()
        .map(std::ffi::OsString::from);
        assert_eq!(
            dyld_vars(vars),
            ["DYLD_FRAMEWORK_PATH", "DYLD_INSERT_LIBRARIES"]
        );
    }

    #[test]
    fn exec_reports_signals_and_timeouts() {
        #[cfg(unix)]
        {
            assert_eq!(sh("kill -9 $$", TIMEOUT).exit, "signal 9");
            let r = sh("sleep 5", Duration::from_millis(200));
            assert_eq!(r.exit, "timed out after 0 s");
        }
        #[cfg(windows)]
        {
            let d = tmp("exec-timeout");
            let shell = PathBuf::from(std::env::var_os("SystemRoot").unwrap())
                .join("System32/WindowsPowerShell/v1.0/powershell.exe");
            let r = System
                .exec(
                    &shell,
                    &s(&["-NoProfile", "-Command", "Start-Sleep -Seconds 5"]),
                    Duration::from_millis(200),
                    &d.join("x.txt"),
                )
                .unwrap();
            assert_eq!(r.exit, "timed out after 0 s");
        }
    }

    #[test]
    fn exec_fails_for_a_missing_program() {
        let d = tmp("exec-missing");
        let e = System
            .exec(&d.join("none"), &[], TIMEOUT, &d.join("x.txt"))
            .unwrap_err();
        assert!(e.contains("none"));
    }
}

"""Coverage of the library by the suite.

  python3 -m pgtest cov [-k PATTERN] [--setup S] [-j N] [--tag T]
                                          build, run, merge, report
  python3 -m pgtest cov --per-case [...]  the same, with one profile set per
                                          case and the line -> cases map
  python3 -m pgtest cov --report-only     report the last run of --tag
  python3 -m pgtest cov --show FILE       lines of a source file (a path
                                          suffix, e.g. fifo/fifo.rs) that no
                                          case reached
  python3 -m pgtest cov --files PATTERNS  report only files matching these

The coverage build is build/cov: a dist tree, written by `cargo xtask dist
--out build/cov/dist` from its own target directory (build/cov/target) under
`cargo llvm-cov show-env`, which instruments the workspace's crates
(-Cinstrument-coverage) and no dependency. The runner loads its dylib, whose
counters the report reads. It is never a build under test: `make test` runs
and `make bless` records from target/dist. Continuous mode needs the counter sections page-aligned
(what clang's driver adds for -fprofile-instr-generate), and rustc links
through clang: CCC_OVERRIDE_OPTIONS appends the alignment, and Rust's
profiler runtime, as -Wl, arguments (ccc_override), to every clang the build
runs.

Continuous mode (%c) is required: pgrun ends with _exit, so a profile
written at exit would be empty. With %c every process's counters are its
file from the start, and survive a crash. Isolated tasks run in the process
that made them, so each case's process is its one profile. The profiles are
merged and exported with Rust's llvm-tools, which read rustc's profiles.
Reports name files relative to the repository (crates/...); files outside it
(the standard library, the dependencies) are left out.

--per-case gives each case its own profiles. It writes cases.json (each
case's window, outcome and whether it matched its expected file on the
coverage build) and map.json ({file: {line: [case ids]}}, load_map): the
cases that reach each line.
"""
import collections
import concurrent.futures as cf
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time

from .cli import LIB_BIN, REPO, SUITE, build_runner, read_expected, run_all, run_case, select
from .registry import load

COV = os.path.join(REPO, "build", "cov")
COV_TARGET = os.path.join(COV, "target")
# Real, as pgrun compares the library's real path with the tree's.
COV_DIST = os.path.join(os.path.realpath(COV), "dist")
COV_ROOT = os.path.join(SUITE, "cov")
COV_DIR = PROFILES = CASES = None
BINARIES = (("host", LIB_BIN),)
REPO_PREFIX = os.path.realpath(REPO) + "/"
# Profiler runtime link and page-aligned counters for clang's links
# (ccc_override).
SECTALIGN = ["__llvm_prf_cnts", "__llvm_prf_bits", "__llvm_prf_data"]


def set_tag(tag):
    global COV_DIR, PROFILES, CASES
    COV_DIR = os.path.join(COV_ROOT, tag)
    PROFILES = os.path.join(COV_DIR, "profiles")
    CASES = os.path.join(COV_DIR, "cases")


def profdata(name):
    return os.path.join(COV_DIR, name + ".profdata")


# ---- the toolchain ----

_sysroot = None


def rust_lib():
    """The toolchain's aarch64-apple-darwin library directory."""
    global _sysroot
    if _sysroot is None:
        _sysroot = subprocess.run(["rustc", "--print", "sysroot"], cwd=REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
    return os.path.join(_sysroot, "lib", "rustlib", "aarch64-apple-darwin")


def llvm(tool):
    """Rust's llvm-tools (llvm-profdata, llvm-cov): they read the profiles
    rustc's runtime writes, which Xcode's may not."""
    p = os.path.join(rust_lib(), "bin", tool)
    if not os.path.exists(p):
        raise SystemExit("%s not found: rustup component add llvm-tools" % p)
    return p


def parse_show_env(text):
    """The variables of `cargo llvm-cov show-env --sh`'s export lines."""
    env = {}
    for line in text.splitlines():
        m = re.match(r"export ([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
        if m:
            v = shlex.split(m.group(2))
            env[m.group(1)] = v[0] if v else ""
    return env


def ccc_override(runtime):
    """CCC_OVERRIDE_OPTIONS: append the profiler runtime and the section
    alignment to every clang command (a leading + appends an argument). The
    runtime goes through -Wl, as the linker's input: a bare .rlib appended to
    a compile that passes -xobjective-c (objc2-exception-helper's build
    script, on a cold target directory) is taken for a second source, and
    clang stops ("cannot specify -o when generating multiple output
    files")."""
    edits = ["+-Wl," + runtime] + ["+-Wl,-sectalign,__DATA,%s,0x4000" % s for s in SECTALIGN]
    return " ".join(edits)


def build_env():
    """The environment of the instrumented build."""
    env = dict(os.environ, CARGO_TARGET_DIR=COV_TARGET)
    out = subprocess.run(["cargo", "llvm-cov", "show-env", "--sh", "--no-cfg-coverage"], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout
    env.update(parse_show_env(out))
    rts = sorted(glob.glob(os.path.join(rust_lib(), "lib", "libprofiler_builtins-*.rlib")))
    if not rts:
        raise SystemExit("no libprofiler_builtins in %s" % rust_lib())
    env["CCC_OVERRIDE_OPTIONS"] = ccc_override(rts[0])
    return env


# ---- the build ----

def build_cov():
    import fcntl
    os.makedirs(COV_ROOT, exist_ok=True)
    with open(os.path.join(COV_ROOT, ".build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _build_cov()


def _build_cov():
    os.makedirs(COV, exist_ok=True)
    log = os.path.join(COV, "build.log")
    with open(log, "w") as f:
        r = subprocess.run(["cargo", "xtask", "dist", "--out", COV_DIST], cwd=REPO,
                           env=build_env(), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit("the coverage build failed; see %s" % log)


def objects(dist=None):
    d = dist or COV_DIST
    return {name: os.path.join(d, rel) for name, rel in BINARIES}


# ---- profiles ----

def nonempty(paths):
    return [p for p in paths if os.path.getsize(p) > 0]


def merge_into(raws, out):
    """Merges raws into the profile out; False when there is none."""
    if not raws:
        if os.path.exists(out):
            os.remove(out)
        return False
    lst = out + ".list"
    with open(lst, "w") as f:
        f.write("\n".join(raws) + "\n")
    subprocess.run([llvm("llvm-profdata"), "merge", "-sparse", "-f", lst, "-o", out], check=True,
                   capture_output=True)
    os.remove(lst)
    return True


def merge(host_raws):
    """The library's profiles, merged."""
    merge_into(host_raws, profdata("host"))


def rel(path):
    """A source path relative to the repository, or None outside it. A
    relative one is the workspace's, which is the repository's root."""
    p = os.path.realpath(path if os.path.isabs(path) else os.path.join(REPO, path))
    return p[len(REPO_PREFIX):] if p.startswith(REPO_PREFIX) else None


def lcov_lines(obj, prof):
    """{file: {line: count}} of one binary's export (llvm-cov's line counts),
    for files in the repository."""
    out = subprocess.run([llvm("llvm-cov"), "export", "-format=lcov", "-skip-functions",
                          "-instr-profile", prof, obj], capture_output=True, text=True, check=True).stdout
    lines = collections.defaultdict(dict)
    cur = None
    for l in out.splitlines():
        if l.startswith("SF:"):
            cur = rel(l[3:])
        elif l.startswith("DA:") and cur:
            n, c = l[3:].split(",")[:2]
            n, c = int(n), int(c)
            lines[cur][n] = max(lines[cur].get(n, 0), c)
    return lines


def union_lines(profs, dist=None):
    """{file: {line: count}} of the binaries that have a profile in profs
    ({binary: profdata path}): the maximum where more than one reaches a
    line."""
    objs = objects(dist)
    out = collections.defaultdict(dict)
    for name, prof in profs.items():
        if not prof or not os.path.exists(prof):
            continue
        for f, ls in lcov_lines(objs[name], prof).items():
            d = out[f]
            for n, c in ls.items():
                d[n] = max(d.get(n, 0), c)
    return out


def summaries(prof, obj):
    """{file: {functions, regions, branches: [covered, count]}} of one
    binary, for files in the repository."""
    data = json.loads(subprocess.run([llvm("llvm-cov"), "export", "-summary-only", "-instr-profile", prof,
                                      obj], capture_output=True, text=True, check=True).stdout)
    out = {}
    for f in data["data"][0]["files"]:
        name = rel(f["filename"])
        if name:
            s = f["summary"]
            out[name] = {k: [s[k]["covered"], s[k]["count"]] for k in ("functions", "regions", "branches")}
    return out


def current_profs():
    return {name: profdata(name) for name, _ in BINARIES}


# ---- reports ----

def crate_of(path):
    parts = path.split("/")
    return "/".join(parts[:2]) if parts[0] == "crates" and len(parts) > 1 else parts[0]


def report_text(lines, patterns=None):
    """Lines covered per file, per crate and in total, as a table."""
    pats = [p for p in (patterns or "").split(",") if p]
    rows = []
    by_crate = collections.defaultdict(lambda: [0, 0])
    total = [0, 0]
    for f in sorted(lines):
        ls = lines[f]
        cov, cnt = sum(1 for c in ls.values() if c > 0), len(ls)
        by_crate[crate_of(f)][0] += cov
        by_crate[crate_of(f)][1] += cnt
        total[0] += cov
        total[1] += cnt
        if not pats or any(p in f for p in pats):
            rows.append((f, cov, cnt))
    pct = lambda c, n: 100.0 * c / n if n else 100.0
    width = max([len(r[0]) for r in rows] + [len("TOTAL"), 40])
    out = ["%-*s %9s %9s %8s" % (width, "file", "covered", "lines", "cover")]
    for f, c, n in rows:
        out.append("%-*s %9d %9d %7.2f%%" % (width, f, c, n, pct(c, n)))
    out.append("")
    for k in sorted(by_crate):
        c, n = by_crate[k]
        out.append("%-*s %9d %9d %7.2f%%" % (width, k, c, n, pct(c, n)))
    out.append("%-*s %9d %9d %7.2f%%" % (width, "TOTAL", total[0], total[1], pct(*total)))
    return "\n".join(out) + "\n"


def report(patterns=None):
    lines = union_lines(current_profs())
    if not lines:
        raise SystemExit("no profile to report in %s" % COV_DIR)
    text = report_text(lines, patterns)
    if not patterns:
        with open(os.path.join(COV_DIR, "report.txt"), "w") as f:
            f.write(text)
    print(text, end="")
    return lines


def export_json(lines):
    """Per file: lines [covered, count], and the functions, regions and
    branches."""
    out = {f: {"lines": [sum(1 for c in ls.values() if c > 0), len(ls)]} for f, ls in lines.items()}
    objs = objects()
    for name, prof in current_profs().items():
        if os.path.exists(prof):
            for f, s in summaries(prof, objs[name]).items():
                out.setdefault(f, {"lines": [0, 0]})[name] = s
    return out


def show(suffix):
    lines = union_lines(current_profs())
    match = sorted(f for f in lines if f.endswith(suffix))
    if not match:
        raise SystemExit("no covered source file ends with " + suffix)
    f = match[0]
    src = open(os.path.join(REPO, f), errors="replace").read().splitlines()
    print(f)
    for n in sorted(lines[f]):
        if lines[f][n] == 0:
            print("%6d  %s" % (n, src[n - 1] if 0 < n <= len(src) else ""))


# ---- runs ----

def safe(case_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)


def run_whole(cases, runner, j, keep=False):
    """Every case, one host profile per process, all in PROFILES (added to
    the last run's with keep); returns every profile there."""
    if not keep:
        shutil.rmtree(PROFILES, ignore_errors=True)
    os.makedirs(PROFILES, exist_ok=True)
    env = {"LLVM_PROFILE_FILE": os.path.join(PROFILES, "host-%p%c.profraw")}

    def one(case):
        obs, _ = run_case(case, runner, tag=".cov", env_extra=env)
        return case, obs[-1]

    for case, outcome in run_all(cases, one, j):
        if outcome.get("kind") in ("error", "timeout"):
            print("%s: %s" % (case.id, outcome))
    return nonempty(glob.glob(os.path.join(PROFILES, "*.profraw")))


def run_per_case(cases, runner, j):
    """Every case with its own host profiles (CASES/<case>/), its window
    recorded; returns the records."""
    shutil.rmtree(CASES, ignore_errors=True)
    os.makedirs(CASES)
    recs = {}
    lock = threading.Lock()
    done = [0]
    t_all = time.time()

    def one(case):
        d = os.path.join(CASES, safe(case.id))
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        obs, _ = run_case(case, runner, tag=".cov",
                          env_extra={"LLVM_PROFILE_FILE": os.path.join(d, "host-%p%c.profraw")})
        t1 = time.time()
        exp = case.expected_path()
        same = os.path.exists(exp) and read_expected(exp)[1] == obs
        return case, t0, t1, obs[-1], same

    for case, t0, t1, outcome, same in run_all(cases, one, j):
        recs[case.id] = {"id": case.id, "name": case.name, "module": case.s.module, "setup": case.setup,
                         "exclusive": case.s.exclusive, "t0": t0, "t1": t1, "dur": round(t1 - t0, 2),
                         "outcome": outcome.get("kind"), "cov_matches_expected": same}
        with lock:
            done[0] += 1
            if not same:
                print("cov: %s differs on the coverage build (%s)" % (case.id, outcome.get("kind")), flush=True)
            if done[0] % 100 == 0:
                print("cov: %d/%d cases, %.0f s" % (done[0], len(cases), time.time() - t_all), flush=True)
    return recs


def build_map(recs, j):
    """Each case's lines, inverted into map.json; the records into
    cases.json."""

    def one(i):
        d = os.path.join(CASES, safe(i))
        host = nonempty(glob.glob(os.path.join(d, "*.profraw")))
        prof = os.path.join(d, "host.profdata")
        return i, union_lines({"host": prof} if merge_into(host, prof) else {})

    m = collections.defaultdict(lambda: collections.defaultdict(list))
    # Each export reads a whole binary: at most one per CPU.
    with cf.ThreadPoolExecutor(max(1, min(j, os.cpu_count() or 1))) as ex:
        for i, lines in ex.map(one, sorted(recs)):
            for f, ls in lines.items():
                for n, c in ls.items():
                    if c > 0:
                        m[f][n].append(i)
    with open(os.path.join(COV_DIR, "cases.json"), "w") as f:
        json.dump(recs, f, indent=0, sort_keys=True)
    with open(os.path.join(COV_DIR, "map.json"), "w") as f:
        json.dump({k: {str(n): sorted(v) for n, v in sorted(ls.items())} for k, ls in sorted(m.items())}, f)
    print("cov: map of %d cases, %d files" % (len(recs), len(m)), flush=True)


def load_map(tag="all"):
    """map.json of a --per-case run: {file: {line: [case ids]}}."""
    with open(os.path.join(COV_ROOT, tag, "map.json")) as f:
        m = json.load(f)
    return {f: {int(n): v for n, v in ls.items()} for f, ls in m.items()}


def cmd_cov(args):
    set_tag(args.tag)
    if args.per_case and args.keep:
        raise SystemExit("--per-case runs every case afresh; it takes no --keep")
    if args.show:
        show(args.show)
        return 0
    if not args.report_only:
        from .cli import write_generated
        build_cov()
        os.makedirs(COV_DIR, exist_ok=True)
        runner = build_runner(COV_DIST)
        cases = select(load(args.k), args.k, args.setup)
        write_generated(cases)
        if args.per_case:
            recs = run_per_case(cases, runner, args.j)
            merge(nonempty(glob.glob(os.path.join(CASES, "*", "*.profraw"))))
            build_map(recs, args.j)
        else:
            merge(run_whole(cases, runner, args.j, args.keep))
    lines = report(args.files or None)
    with open(os.path.join(COV_DIR, "summary.json"), "w") as f:
        json.dump(export_json(lines), f, indent=1, sort_keys=True)
    return 0

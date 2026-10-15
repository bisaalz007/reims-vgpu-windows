"""The orchestrator: generates scenarios, runs them, compares them with
their expected files, and records ("blesses") expected files.

  python3 -m pgtest gen [--check]
  python3 -m pgtest list
  python3 -m pgtest run   [-k PATTERN] [--setup S] [-j N] [--dist DIR]
  python3 -m pgtest bless [-k PATTERN] [--setup S] [-j N] [--runs N] [--update] [--dist DIR]
  python3 -m pgtest cov   [-k PATTERN] [--setup S] [-j N] [--tag T] [--per-case] ...
  python3 -m pgtest diffsum [A [B]] | --dirs OLD NEW   (pgtest/diffsum.py)

Run from tests/suite (the Makefile's `make test` and `make bless` do). The
rules are docs/testing.md's, section 5.7: an expected file records what the
codebase does, from its own build; a person reviews it, CI never writes one,
and it is recorded again only when its input changed, or when `bless
--update` records a deliberate change of behaviour, whose expected-file diff
is reviewed in the same commit.

`run` and `bless` take the build under test: the checkout's dist tree
($PGTEST_DIST, else target/dist, which `cargo xtask dist` writes), or --dist.
The runner links the tree's libreims_vgpu.dylib and loads it from there.
"""
import argparse
import difflib
import fcntl
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from . import normalize
from . import registry
from .registry import SUITE, load, load_all

REPO = os.path.dirname(os.path.dirname(SUITE))
GEN = os.path.join(SUITE, "generated")
EXPECTED = os.path.join(SUITE, "expected")
OUT = os.path.join(SUITE, "out")
BIN = os.path.join(SUITE, "bin")
FIXTURES = os.path.join(SUITE, "fixtures")
RUNNER_SRC = os.path.join(SUITE, "runner", "pgrun.m")
RUNNER_BUILD = os.path.join(SUITE, "runner", "build.sh")
# The build under test: the checkout's dist tree, where `cargo xtask dist`
# writes it in the workspace at the repository root (docs/development.md).
DEFAULT_DIST = os.environ.get("PGTEST_DIST") or os.path.join(REPO, "target", "dist")
# The library the runner loads, in a dist tree.
LIB_BIN = "lib/libreims_vgpu.dylib"


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---- the host key (docs/testing.md, section 5.7) ----

_host_key = None


def host_key():
    """GPU model, core count and macOS build: what host-reporting values and
    the pixels the library's shaders draw depend on."""
    global _host_key
    if _host_key is None:
        io = subprocess.run(["ioreg", "-rd1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        model = re.search(r'"model" = "([^"]+)"', io)
        cores = re.search(r'"gpu-core-count" = (\d+)', io)
        build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
        m = re.sub(r"[^A-Za-z0-9]", "", model.group(1)) if model else "unknown"
        _host_key = "%s-%sc-%s" % (m, cores.group(1) if cores else "0", build)
    return _host_key


# ---- the runner ----

def runner_version():
    """What the output depends on besides the scenario and the library."""
    hsh = hashlib.sha256()
    for p in (RUNNER_SRC, normalize.__file__):
        hsh.update(open(p, "rb").read())
    return hsh.hexdigest()[:16]


# The dist tree each runner was built against (build_runner), which the
# runner is told with --build: it reports the library as "<build>" when it
# comes from there.
BUILDS = {}


def build_runner(dist):
    """The runner for the dist tree `dist`, built again when pgrun.m or
    build.sh changed. It loads the tree's dylib at run time, so a new build of
    the library needs no new runner."""
    dist = os.path.realpath(dist)
    if not os.path.isfile(os.path.join(dist, LIB_BIN)):
        raise SystemExit("%s: no %s: run `cargo xtask dist`" % (dist, LIB_BIN))
    tag = hashlib.sha256(dist.encode()).hexdigest()[:10]
    out = os.path.join(BIN, "pgrun-" + tag)
    stamp = out + ".src"
    want = sha256_file(RUNNER_SRC) + " " + sha256_file(RUNNER_BUILD)
    if not os.path.exists(out) or not os.path.exists(stamp) or open(stamp).read() != want:
        subprocess.run([RUNNER_BUILD, dist, out], check=True)
        open(stamp, "w").write(want)
    BUILDS[out] = dist
    return out


# ---- cases ----

class Case:
    def __init__(self, s, setup):
        self.s = s
        self.setup = setup
        self.name = s.name
        self.id = "%s[%s]" % (s.name, setup)

    def gen_path(self):
        return os.path.join(GEN, self.s.module, self.name + ".jsonl")

    def expected_path(self):
        base = self.setup + ("@" + host_key() if self.s.host else "")
        return os.path.join(EXPECTED, self.s.module, self.name, base + ".jsonl")

    def input_hash(self):
        with open(self.gen_path(), "rb") as f:
            ops = b"".join(l for l in f.read().splitlines(keepends=True) if not l.startswith(b"#"))
        hsh = hashlib.sha256(ops)
        hsh.update(runner_version().encode())
        return hsh.hexdigest()


class BlessRun:
    """Run i of a case being blessed: scheduled as the case is (run_all)."""
    def __init__(self, case, i):
        self.case = case
        self.i = i
        self.round = i
        self.s = case.s
        self.setup = case.setup
        self.name = case.name
        self.id = case.id

    def expected_path(self):
        return self.case.expected_path()


def select(scenarios, pattern, setup):
    """The cases of the scenarios matching pattern, in the setups listed
    (all if none). A chain (map-isolate, a setup a phase) is selected when
    it is listed itself or every setup in it is."""
    cases = []
    pats = pattern.split(",") if pattern else None
    want = setup.split(",") if setup else None
    for name in sorted(scenarios):
        s = scenarios[name]
        if pats and not registry.matches(name, s.module, pats):
            continue
        for st in s.setups:
            if want and st not in want and not all(x in want for x in st.split("-")):
                continue
            cases.append(Case(s, st))
    return cases


def _run_process(runner, gen, log, err, setup, watchdog, env, extra=()):
    t = time.time()
    with open(err, "w") as ef:
        build = ["--build", BUILDS[runner]] if runner in BUILDS else []
        p = subprocess.Popen([runner, gen, log, "--setup", setup, "--fixtures", FIXTURES] + build + list(extra),
                             stdout=ef, stderr=ef, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=watchdog + 30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = None
    return normalize.normalize(log, rc, time.time() - t)


def run_case(case, runner, tag="", env_extra=None):
    """Runs one case in its own process; returns (normalized observations,
    raw log path).

    A scenario in phases runs a process a phase, in the phase's setup, in
    order, sharing a handoff directory this function owns (a temporary
    directory, removed after). Its observations are the phases' in order,
    each record tagged "phase": k, each phase ending in its own outcome; a
    phase that does not complete ends the case. The log returned is the
    last phase's."""
    d = os.path.join(OUT, case.s.module)
    os.makedirs(d, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYLD_") and k != "MTL_DEBUG_LAYER"}
    env.update(env_extra or {})
    base = os.path.join(d, "%s.%s%s" % (case.name, case.setup, tag))
    setups = case.s.phase_setups(case.setup)
    if len(setups) == 1:
        watchdog = case.s.ops[0].get("watchdog_s", 30)
        return _run_process(runner, case.gen_path(), base + ".jsonl", base + ".stderr", case.setup,
                            watchdog, env), base + ".jsonl"
    configs = [case.s.ops[0]] + [case.s.ops[i + 1] for i, o in enumerate(case.s.ops) if o.get("op") == "phase"]
    handoff = tempfile.mkdtemp(prefix="pgtest-handoff-")
    obs = []
    try:
        for k, setup in enumerate(setups):
            log = "%s.phase%d.jsonl" % (base, k)
            got = _run_process(runner, case.gen_path(), log, "%s.phase%d.stderr" % (base, k), setup,
                               configs[k].get("watchdog_s", 30), env,
                               ["--phase", str(k), "--handoff", handoff])
            for o in got:
                o["phase"] = k
            obs += got
            if got[-1].get("kind") != "completed":
                break
    finally:
        shutil.rmtree(handoff, ignore_errors=True)
    return obs, log


# ---- scheduling ----

def default_jobs():
    """Cases at once. A case is latency-bound, not CPU-bound: the whole
    suite spends about 4% of its wall time on a CPU (runner and services;
    the rest is waits, settle windows and deadlines that are meant to
    pass), so twice the CPUs keeps every core busy without starving a
    case (docs/testing.md, section 5.4)."""
    return int(os.environ.get("PGTEST_JOBS", "0")) or max(8, 2 * (os.cpu_count() or 4))


DURATIONS = os.path.join(OUT, "durations.json")


def load_durations():
    """The wall time of each case's last run (seconds, by case id): the
    orchestrator starts the longest first, so no long case starts last."""
    try:
        with open(DURATIONS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_durations(new):
    """Merges this run's durations into the cache, under a lock: several
    orchestrators may finish at once."""
    if not new:
        return
    os.makedirs(OUT, exist_ok=True)
    with open(DURATIONS + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = load_durations()
        d.update({k: round(v, 3) for k, v in new.items()})
        tmp = DURATIONS + ".tmp%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(d, f, sort_keys=True, indent=0)
        os.replace(tmp, DURATIONS)


def static_estimate(case):
    """A case's wall time before it has a recorded one, from its expected
    file: the waits it expects to time out burn their deadline, a hang its
    watchdog, and each checkpoint settles for its quiet window."""
    try:
        _, obs = read_expected(case.expected_path())
    except (OSError, ValueError):
        return 1.0
    phases = [[]]
    for o in case.s.ops:
        if o.get("op") == "phase":
            phases.append([])
        else:
            phases[-1].append(o)
    t = 0.1 * len(phases)
    for o in obs:
        ph = phases[min(o.get("phase", 0), len(phases) - 1)]
        i = o.get("i")
        op = ph[i] if isinstance(i, int) and 0 <= i < len(ph) else {}
        if o.get("ev") == "wait" and not o.get("ok"):
            t += op.get("timeout_ms", 5000) / 1000
        elif o.get("ev") == "events":
            t += op.get("quiet_ms", 40) / 1000 + 0.02
        elif o.get("ev") == "outcome" and o.get("kind") == "watchdog":
            t += ph[0].get("watchdog_s", 30) if ph else 30
    return t


_iokit = None


def iosurface_clients():
    """The IOSurface clients open machine-wide (the kernel allows about 1024; a
    runner holds one, and its isolated tasks, which run in its process, open
    none of their own), or None if IOKit cannot say. Counted as `ioreg -c
    IOSurfaceRootUserClient` does, from IOSurfaceRoot's children, in well under
    a millisecond."""
    global _iokit
    try:
        import ctypes
        if _iokit is None:
            k = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
            k.IOServiceMatching.restype = ctypes.c_void_p
            k.IOServiceMatching.argtypes = [ctypes.c_char_p]
            k.IOServiceGetMatchingService.restype = ctypes.c_uint
            k.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            k.IORegistryEntryGetChildIterator.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint)]
            k.IOIteratorNext.restype = ctypes.c_uint
            k.IOIteratorNext.argtypes = [ctypes.c_uint]
            k.IOObjectRelease.argtypes = [ctypes.c_uint]
            k.IOObjectConformsTo.argtypes = [ctypes.c_uint, ctypes.c_char_p]
            _iokit = k
        k = _iokit
        root = k.IOServiceGetMatchingService(0, k.IOServiceMatching(b"IOSurfaceRoot"))
        if not root:
            return None
        it = ctypes.c_uint()
        try:
            if k.IORegistryEntryGetChildIterator(root, b"IOService", ctypes.byref(it)):
                return None
            n = 0
            while True:
                o = k.IOIteratorNext(it.value)
                if not o:
                    break
                if k.IOObjectConformsTo(o, b"IOSurfaceRootUserClient"):
                    n += 1
                k.IOObjectRelease(o)
            k.IOObjectRelease(it.value)
            return n
        finally:
            k.IOObjectRelease(root)
    except (OSError, AttributeError):
        return None


# A case starts only while fewer IOSurface clients than this are open
# machine-wide (the kernel allows about 1024; each runner holds one).
IOSURFACE_HIGH = int(os.environ.get("PGTEST_IOSURFACE_HIGH", "600"))
# The scenarios that declare many isolated tasks (Scenario.services) take
# them from this budget while they run. It bounds no IOSurface client: an
# isolated task's service runs on threads of the runner's process and opens
# none. What it does is space out the four cases that define tasks 1-255
# under isolation (interplay_limits_all_tasks and
# interplay_limits_root_range_full, interplay_servicewalk_31 and 32), each of
# which ramps up 255 services at once, every one with serial queues and two
# Metal command queues of its own. 300 is at least 255, so one of them
# starts, and below 510, so a second waits for it to end, while a case of up
# to 45 tasks still runs beside it. interplay_limits_all_tasks[isolate] is
# load-sensitive (README.md). A case that declares more than the budget
# starts only when the whole budget is free.
SERVICE_BUDGET = int(os.environ.get("PGTEST_SERVICE_BUDGET", "300"))


def services_of(case):
    """The services a case may hold at once: one a task, only under
    isolation."""
    return getattr(case.s, "services", 0) if "isolate" in case.setup else 0


def run_all(cases, one, j, durations=None):
    """Yields one(case) for every case, as each finishes.

    The shared cases run j at a time, the longest first (by their last
    recorded time, else static_estimate). A case starts only while the
    machine's IOSurface clients are below IOSURFACE_HIGH, and those that
    declare many services take them from SERVICE_BUDGET. Then the exclusive
    ones (those that fault or hang the GPU) run one at a time, once every
    shared case has ended. Each case's wall time is recorded in durations (a dict) when one is
    given, and merged into the cache at the end."""
    if durations is None:
        durations = {}
    try:
        yield from _run_all(cases, one, j, durations)
    finally:
        save_durations(durations)


def _run_all(cases, one, j, durations):
    known = load_durations()
    est = {}
    for c in cases:
        est[c.id] = known[c.id] if c.id in known else static_estimate(c)
    # Longest first, within rounds (a blessed case's first runs, then its
    # second...).
    shared = sorted((c for c in cases if not c.s.exclusive), key=lambda c: (getattr(c, "round", 0), -est[c.id]))
    alone = [c for c in cases if c.s.exclusive]

    def timed(c):
        t = time.time()
        try:
            return one(c)
        finally:
            durations[c.id] = max(durations.get(c.id, 0), time.time() - t)

    cv = threading.Condition()
    pending = list(shared)
    budget = [SERVICE_BUDGET]
    done = queue.Queue()

    def take():
        """The next case this worker may start, or None when none is left."""
        with cv:
            while True:
                if not pending:
                    return None
                for k, c in enumerate(pending):
                    w = services_of(c)
                    if w <= budget[0] or w > SERVICE_BUDGET and budget[0] == SERVICE_BUDGET:
                        budget[0] -= min(w, SERVICE_BUDGET)
                        return pending.pop(k)
                cv.wait(0.5)

    def give(c):
        with cv:
            budget[0] += min(services_of(c), SERVICE_BUDGET)
            cv.notify_all()

    def worker():
        while True:
            c = take()
            if c is None:
                break
            while True:
                n = iosurface_clients()
                if n is None or n < IOSURFACE_HIGH:
                    break
                time.sleep(0.25)
            try:
                r = (True, timed(c))
            except BaseException as e:  # handed to the caller
                r = (False, e)
            give(c)
            done.put(r)

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(j, len(shared))))]
    for t in ths:
        t.start()
    for _ in shared:
        ok, r = done.get()
        if not ok:
            raise r
        yield r
    for t in ths:
        t.join()
    for c in alone:
        yield timed(c)


# ---- commands ----

def cmd_gen(args):
    scenarios = load_all()
    want = {}
    for s in scenarios.values():
        want[os.path.join(GEN, s.module, s.name + ".jsonl")] = s.jsonl()
    stale, changed = [], []
    for root, _, files in os.walk(GEN):
        if os.path.basename(root) in registry.FAILED:
            continue
        for f in files:
            p = os.path.join(root, f)
            if p not in want:
                stale.append(p)
    for p, text in want.items():
        old = open(p).read() if os.path.exists(p) else None
        if old != text:
            changed.append(p)
    if args.check:
        for p in changed:
            print("out of date:", os.path.relpath(p, SUITE))
        for p in stale:
            print("stale:", os.path.relpath(p, SUITE))
        return 1 if changed or stale else 0
    for p in stale:
        os.remove(p)
    for p in changed:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(want[p])
    print("%d scenarios, %d cases; %d written, %d removed" % (
        len(scenarios), sum(len(s.setups) for s in scenarios.values()), len(changed), len(stale)))
    return 0


def write_generated(cases):
    """Writes the generated files of the selected cases only, so that runs
    over different modules at once never write each other's files."""
    done = set()
    for c in cases:
        p = c.gen_path()
        if p in done:
            continue
        done.add(p)
        text = c.s.jsonl()
        if not os.path.exists(p) or open(p).read() != text:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp%d" % os.getpid()
            open(tmp, "w").write(text)
            os.replace(tmp, p)


def cmd_list(args):
    for name, s in sorted(load_all().items()):
        print("%-48s %-14s %s%s" % (name, s.module, ",".join(s.setups), " (host)" if s.host else ""))
    return 0


def read_expected(path):
    with open(path) as f:
        header = json.loads(f.readline())
        return header, [json.loads(l) for l in f if l.strip()]


def show(o):
    return json.dumps(o, sort_keys=True)


def run_one_and_compare(case, runner, tag=""):
    """Runs case once and compares it with its expected file: (case,
    status, detail, observations, expected observations), status pass, fail
    or missing."""
    exp = case.expected_path()
    if not os.path.exists(exp):
        return case, "missing", None, None, None
    obs, log = run_case(case, runner, tag=tag)
    header, want = read_expected(exp)
    if header.get("input") != case.input_hash():
        return case, "fail", "the expected file was recorded from other input: bless it again", obs, None
    if obs == want:
        return case, "pass", None, obs, want
    d = "\n".join(difflib.unified_diff([show(o) for o in want], [show(o) for o in obs],
                                       "expected", "observed (%s)" % os.path.relpath(log, SUITE), lineterm="", n=2))
    return case, "fail", d, obs, want


def run_and_compare(cases, runner, j):
    """Runs every case and compares it with its expected file; yields
    run_one_and_compare's tuple as each finishes."""
    yield from run_all(cases, lambda case: run_one_and_compare(case, runner), j)


def cmd_run(args):
    scenarios = load(args.k)
    cases = select(scenarios, args.k, args.setup)
    write_generated(cases)
    runner = build_runner(args.dist)
    results = {"pass": [], "fail": [], "missing": []}

    for case, status, detail, _, _ in run_and_compare(cases, runner, args.j):
        results[status].append(case)
        if status != "pass" or args.verbose:
            print("%-4s %s" % (status.upper(), case.id), flush=True)
        if detail:
            print("\n".join(detail.splitlines()[:args.max_diff]), flush=True)
    print("%d passed, %d failed, %d missing (host %s)" % (
        len(results["pass"]), len(results["fail"]), len(results["missing"]), host_key()))
    return 0 if not results["fail"] and not results["missing"] else 1


def cmd_bless(args):
    scenarios = load(args.k)
    cases = select(scenarios, args.k, args.setup)
    write_generated(cases)
    print("bless: recording from %s" % args.dist, flush=True)
    runner = build_runner(args.dist)
    counts = {}
    for case, status, detail in bless_cases(cases, runner, args.runs, args.j, counts, args.update):
        if status != "unchanged":
            print("%-16s %s" % (status.upper(), case.id), flush=True)
        if detail:
            print("\n".join(detail.splitlines()[:args.max_diff]), flush=True)
    print(", ".join("%d %s" % (v, k) for k, v in sorted(counts.items())))
    return 0 if not counts.get("refused") and not counts.get("nondeterministic") else 1


def bless_cases(cases, runner, nruns, j, counts=None, update=False):
    """Records the expected file of every case whose file is missing or was
    recorded from other input, or of every case with update, from nruns runs
    that must agree and complete; yields (case, status, detail) per case
    run, status blessed, refused, nondeterministic, or unchanged when update
    recorded what the file already held, and counts every status (the cases
    not run as unchanged too) in counts."""
    if counts is None:
        counts = {}
    todo = []
    for case in cases:
        exp = case.expected_path()
        if not update and os.path.exists(exp) and read_expected(exp)[0].get("input") == case.input_hash():
            counts["unchanged"] = counts.get("unchanged", 0) + 1
        else:
            todo.append(case)
    # Each of a case's runs is a task of its own, in its own process: the
    # first runs of every case, then the second, then the third, so that a
    # case's runs meet different moments of the whole run rather than one.
    tasks = [BlessRun(c, i) for i in range(nruns) for c in todo]
    # What a case's runs so far agree on, kept as text: the first run's
    # observations, and the first difference from them.
    runs = {}

    def one(run):
        obs, log = run_case(run.case, runner, tag=".bless%d" % run.i)
        return run, obs, log

    def judged():
        for run, obs, log in run_all(tasks, one, j):
            case = run.case
            r = runs.setdefault(case.id, {"n": 0, "first": None, "why": None, "diff": None})
            r["n"] += 1
            text = [show(o) for o in obs]
            if obs[-1].get("kind") in ("error", "timeout") and not r["why"]:
                r["why"] = "outcome %s: %s" % (obs[-1].get("kind"), log)
            if r["first"] is None:
                r["first"] = text
            elif text != r["first"] and not r["diff"]:
                r["diff"] = "\n".join(difflib.unified_diff(r["first"], text, "run 0", "run n", lineterm="", n=1))
            if r["n"] < nruns:
                continue
            del runs[case.id]
            if r["why"]:
                yield case, "refused", r["why"]
            elif r["diff"]:
                yield case, "nondeterministic", r["diff"]
            else:
                exp = case.expected_path()
                text = show({"expected": case.name, "setup": case.setup,
                             "host": host_key() if case.s.host else None, "input": case.input_hash()}) + "\n"
                text += "".join(line + "\n" for line in r["first"])
                if os.path.exists(exp) and open(exp).read() == text:
                    # --update recorded what the file already holds.
                    yield case, "unchanged", None
                    continue
                os.makedirs(os.path.dirname(exp), exist_ok=True)
                with open(exp, "w") as f:
                    f.write(text)
                yield case, "blessed", None

    print("bless: %d cases, %d unchanged; %d runs to make" % (len(cases), counts.get("unchanged", 0), len(tasks)),
          flush=True)
    for case, status, detail in judged():
        counts[status] = counts.get(status, 0) + 1
        yield case, status, detail


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pgtest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--check", action="store_true")
    sub.add_parser("list")
    for name in ("run", "bless"):
        p = sub.add_parser(name)
        p.add_argument("-k", default="", help="scenario or module names, fnmatch patterns, comma-separated")
        p.add_argument("--setup", default="", help="blocks,map,isolate")
        p.add_argument("-j", type=int, default=default_jobs())
        p.add_argument("--dist", default=DEFAULT_DIST,
                       help="the build under test, a dist tree (default: $PGTEST_DIST, else target/dist)")
        p.add_argument("--max-diff", type=int, default=40)
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "bless":
            p.add_argument("--runs", type=int, default=3, help="runs that must agree before a case is blessed")
            p.add_argument("--update", action="store_true",
                           help="record the selected cases again even when their input is unchanged: for a "
                                "deliberate change of behaviour, whose expected-file diff is reviewed in the "
                                "same commit")
    c = sub.add_parser("cov")
    c.add_argument("-k", default="")
    c.add_argument("--setup", default="")
    c.add_argument("-j", type=int, default=default_jobs())
    c.add_argument("--report-only", action="store_true")
    c.add_argument("--keep", action="store_true", help="add to the last run's profiles")
    c.add_argument("--show", default="", help="uncovered lines of the source file with this suffix")
    c.add_argument("--tag", default="all", help="names this run's profile directory, cov/TAG")
    c.add_argument("--files", default="", help="report only source files whose path contains one of these, comma-separated")
    c.add_argument("--per-case", action="store_true", help="each case's own profiles, and map.json: line -> cases")
    d = sub.add_parser("diffsum", help="summarize a change to expected files by record kind and by scenario")
    d.add_argument("revs", nargs="*", help="revision A (default HEAD) and B (default the working tree)")
    d.add_argument("--dirs", nargs=2, metavar=("OLD", "NEW"), help="compare two result sets instead")
    d.add_argument("-k", default="", help="only these scenarios or modules, fnmatch patterns, comma-separated")
    d.add_argument("--samples", type=int, default=3, help="value changes shown per field")
    d.add_argument("--scenarios", type=int, default=40, help="scenarios listed")
    args = ap.parse_args(argv)
    if args.cmd == "diffsum":
        from .diffsum import cmd_diffsum
        return cmd_diffsum(args)
    if args.cmd == "cov":
        from .coverage import cmd_cov
        return cmd_cov(args)
    return {"gen": cmd_gen, "list": cmd_list, "run": cmd_run, "bless": cmd_bless}[args.cmd](args)

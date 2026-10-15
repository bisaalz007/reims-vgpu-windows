"""Scenario registration: a scenario module declares its scenarios with
@scenario; the orchestrator imports the modules under tests/suite/scenarios
and builds their scenarios.

load_all() imports every module and builds every scenario. load(pattern)
builds only what a selection needs. A scenario's name is known only once its
function has run, so an index records the names each function builds, keyed
by the sources they were built from ("The index", below).
"""
import fnmatch
import hashlib
import importlib
import json
import os
import pkgutil
import sys
import traceback

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY = {}


def scenario(fn=None, **meta):
    """Registers a function that returns a Scenario (or a list of them). The
    scenario's name is its own; a function may build many."""
    def deco(f):
        _REGISTRY[f.__module__ + "." + f.__name__] = (f, meta)
        return f
    return deco(fn) if fn is not None else deco


FAILED = set()  # modules that did not import or build, when not strict


def matches(name, module, pats):
    """Whether any of pats, fnmatch patterns, names a scenario or its module
    (cli.select)."""
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(module, p) for p in pats)


def _module_of(key):
    return key.split(".")[1]


def _module_names():
    """Every module under tests/suite/scenarios, sorted."""
    if SUITE not in sys.path:
        sys.path.insert(0, SUITE)
    import scenarios  # noqa: F401  (tests/suite/scenarios)
    return [m.name for m in sorted(pkgutil.iter_modules(scenarios.__path__), key=lambda m: m.name)]


def _import(mods, strict):
    for m in mods:
        try:
            importlib.import_module("scenarios." + m)
        except Exception:
            if strict:
                raise
            FAILED.add(m)
            sys.stderr.write("pgtest: skipping module %s: it does not import:\n%s" % (m, traceback.format_exc(limit=3)))


def _build(keys, strict, built, out):
    """Runs the functions keys in order into out ({name: Scenario}), refusing
    a name out holds already, and records the names each one built in built
    ({key: [name]}). A module that did not import or build is skipped."""
    for key in keys:
        f, _ = _REGISTRY[key]
        module = _module_of(key)
        if module in FAILED:
            continue
        try:
            got = f()
        except Exception:
            if strict:
                raise
            FAILED.add(module)
            sys.stderr.write("pgtest: skipping module %s: %s does not build:\n%s" % (module, key, traceback.format_exc(limit=4)))
            continue
        built[key] = []
        for s in got if isinstance(got, (list, tuple)) else [got]:
            if s.name in out:
                raise SystemExit("duplicate scenario name %s (%s)" % (s.name, key))
            s.module = module
            out[s.name] = s
            built[key].append(s.name)


def load_all(strict=None):
    """Imports every scenario module and builds every scenario. Returns
    {name: Scenario}, refusing duplicate names.

    Not strict (the default unless PGTEST_STRICT=1), a module that fails to
    import or build is reported and skipped, so that one scenario module in
    progress does not stop work on the others; its generated files are left
    alone. CI runs strict."""
    if strict is None:
        strict = os.environ.get("PGTEST_STRICT") == "1"
    mods = _module_names()
    index = _read_index()
    keys = _module_keys(mods, _scan(index))
    _import(mods, strict)
    built, out = {}, {}
    _build(sorted(_REGISTRY), strict, built, out)
    _save_index(index, mods, keys, built)
    return {name: s for name, s in out.items() if s.module not in FAILED}


def load(pattern="", strict=None):
    """Builds what a selection needs: every scenario whose name or module
    matches one of pattern's comma-separated fnmatch patterns (cli.select),
    and perhaps others. Without a pattern, or without an index, load_all().

    It imports and builds, in load_all's order and under its failure rules:
    - every module whose index entry is missing or keyed by other sources,
      whole, and indexes it again;
    - of the other modules, the functions whose names the index says a
      pattern matches, or all of a module's when a pattern names it.
    Should a function build other names than its entry says, the index is
    not trusted: it builds everything left, as load_all would."""
    if not pattern:
        return load_all(strict)
    if strict is None:
        strict = os.environ.get("PGTEST_STRICT") == "1"
    pats = pattern.split(",")
    mods = _module_names()
    index = _read_index()
    keys = _module_keys(mods, _scan(index))
    fresh = {m: index["modules"][m]["fns"] for m in mods
             if keys[m] and index["modules"].get(m, {}).get("key") == keys[m]}
    if not fresh:
        return load_all(strict)
    stale = [m for m in mods if m not in fresh]
    want = set()
    for m, fns in fresh.items():
        whole = any(fnmatch.fnmatch(m, p) for p in pats)
        want.update(k for k, names in fns.items() if whole or any(matches(n, m, pats) for n in names))
    _import(sorted(set(stale) | {_module_of(k) for k in want}), strict)
    built, out = {}, {}
    _build(sorted(k for k in _REGISTRY if _module_of(k) in stale or k in want), strict, built, out)
    # The index lists every function of a module it holds, and only scenario
    # modules register any.
    trusted = all(k.startswith("scenarios.") for k in _REGISTRY)
    for m in {_module_of(k) for k in want} - FAILED:
        trusted = trusted and sorted(k for k in _REGISTRY if _module_of(k) == m) == sorted(fresh[m])
    trusted = trusted and all(built[k] == fresh[_module_of(k)][k] for k in want if k in built)
    if not trusted:
        sys.stderr.write("pgtest: the scenario index does not match the scenarios: building them all\n")
        _import([m for m in mods if m not in FAILED], strict)
        _build(sorted(k for k in _REGISTRY if k not in built), strict, built, out)
        _save_index(index, mods, keys, built)
        return {name: s for name, s in out.items() if s.module not in FAILED}
    # Refuse a name built twice, as load_all does, at the later function in
    # its order, among the functions this run built and those it did not.
    names = {k: names for fns in fresh.values() for k, names in fns.items()}
    names.update(built)
    seen = set()
    for key in sorted(names):
        if _module_of(key) in FAILED and key not in built:
            continue
        for n in names[key]:
            if n in seen:
                raise SystemExit("duplicate scenario name %s (%s)" % (n, key))
            seen.add(n)
    _save_index(index, mods, keys, {k: v for k, v in built.items() if _module_of(k) in stale})
    return {name: s for name, s in out.items() if s.module not in FAILED}


# ---- the index ----
#
# One file a checkout, index_path(). For each scenario module it holds the
# names each of its functions builds, under a key: the hash of the module's
# source and of every pgtest and scenario module it imports, directly or
# through others, found by parsing their imports. So an edit re-indexes the
# modules built from what it changed, and a selection that meets one builds
# that module whole. The index only chooses which functions a selection
# runs; what a case runs is always built from the sources. Each file's
# imports are kept there too, by the file's hash, so only a changed file is
# parsed again.
#
# It relies on a rule for scenario modules (README.md, "Writing scenarios"):
# a function builds the same scenarios whatever the process built before it.

INDEX_VERSION = 1
_PACKAGES = ("pgtest", "scenarios")


def index_path():
    return os.path.join(os.path.expanduser("~"), ".cache", "pvd-pgtest",
                        "index-%s.json" % hashlib.sha256(SUITE.encode()).hexdigest()[:12])


def _read_index():
    try:
        with open(index_path()) as f:
            index = json.load(f)
        if index.get("version") == INDEX_VERSION and isinstance(index["files"], dict) and all(
                isinstance(e.get("fns"), dict) for e in index["modules"].values()):
            return index
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return {"version": INDEX_VERSION, "files": {}, "modules": {}}


def _imports(rel, source):
    """The files of this suite's packages a source imports (pgtest/x.py,
    scenarios/x.py), anywhere in it, or None if it does not parse."""
    import ast
    try:
        tree = ast.parse(source, rel)
    except (SyntaxError, ValueError):
        return None
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in _PACKAGES and len(parts) > 1:
                    out.add("%s/%s.py" % (parts[0], parts[1]))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                parts = (node.module or "").split(".")
            elif node.level == 1:
                parts = [rel.split("/")[0]] + (node.module.split(".") if node.module else [])
            else:
                continue
            if parts[0] not in _PACKAGES:
                continue
            if len(parts) > 1:
                out.add("%s/%s.py" % (parts[0], parts[1]))
            else:
                out.update("%s/%s.py" % (parts[0], a.name) for a in node.names)
    return sorted(out)


def _scan(index):
    """{file: [sha256, imports]} for every source of the two packages, the
    imports taken from the index where the file is unchanged."""
    files = {}
    for pkg in _PACKAGES:
        d = os.path.join(SUITE, pkg)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"):
                continue
            rel = pkg + "/" + f
            with open(os.path.join(d, f), "rb") as fh:
                source = fh.read()
            sha = hashlib.sha256(source).hexdigest()
            old = index["files"].get(rel)
            files[rel] = old if isinstance(old, list) and old[0] == sha else [sha, _imports(rel, source)]
    return files


def _module_keys(mods, files):
    """{module: the hash of the sources it is built from}, None for a module
    that is not a file, or that imports a file that does not parse."""
    base = ["%d %d.%d" % ((INDEX_VERSION,) + sys.version_info[:2])]
    base += ["%s %s" % (p, files[p][0]) for p in ("pgtest/__init__.py", "scenarios/__init__.py") if p in files]
    keys = {}
    for m in mods:
        root = "scenarios/%s.py" % m
        seen, todo, ok = set(), [root], root in files
        while todo and ok:
            p = todo.pop()
            if p in seen or p not in files:
                continue
            seen.add(p)
            ok = files[p][1] is not None
            todo.extend(files[p][1] or ())
        text = "\n".join(base + ["%s %s" % (p, files[p][0]) for p in sorted(seen)])
        keys[m] = hashlib.sha256(text.encode()).hexdigest() if ok else None
    return keys


def _save_index(index, mods, keys, built):
    """Indexes each module built whole in built ({key: [name]}), under its
    key from before it was imported, unless its sources changed since; drops
    the modules that failed. Never fails: a run without an index is only
    slower."""
    try:
        files = _scan(index)
        now = _module_keys(mods, files)
        fns = {}
        for k, names in built.items():
            fns.setdefault(_module_of(k), {})[k] = names
        modules = {m: e for m, e in index["modules"].items() if m in mods and m not in FAILED}
        for m, got in fns.items():
            modules.pop(m, None)
            whole = sorted(got) == sorted(k for k in _REGISTRY if _module_of(k) == m)
            if m not in FAILED and whole and keys.get(m) and now.get(m) == keys[m]:
                modules[m] = {"key": keys[m], "fns": got}
        new = {"version": INDEX_VERSION, "files": files, "modules": modules}
        if new == index:
            return
        path = index_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "%s.tmp%d" % (path, os.getpid())
        with open(tmp, "w") as f:
            json.dump(new, f, sort_keys=True)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        pass

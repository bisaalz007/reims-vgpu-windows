"""diffsum: a summary of a change to expected files, by record kind and by
scenario, so that a large re-record can be reviewed.

  python3 -m pgtest diffsum [A [B]]            git: the expected files at
                                               revision A (default HEAD)
                                               against revision B (default
                                               the working tree)
  python3 -m pgtest diffsum --dirs OLD NEW     two result sets, each a tree
                                               laid out as expected/ is
  ... [-k PATTERNS] [--samples N] [--scenarios N]

A result set is a directory of <module>/<scenario>/<setup>[@host].jsonl
files, each an expected file or a list of observations without its header
line: expected/ itself, or observations a script wrote in that layout.

Each pair of files is compared record by record: the records the two share
in order are matched (difflib), and a record that sits where one of the same
kind sat in the other file is compared field by field. The summary counts:
- per record kind (the "ev" of a record, "api:<name>" for an api call): the
  records added, removed and changed, and for the changed ones each field
  added (+), removed (-) or changed (~), with the commonest value changes;
- per scenario: the files and records changed, and the kinds;
- the files added, removed and changed, and those whose header's input hash
  alone changed.
"""
import collections
import difflib
import json
import os
import subprocess

from .registry import SUITE, matches

REPO = os.path.dirname(os.path.dirname(SUITE))
EXPECTED_REL = os.path.relpath(os.path.join(SUITE, "expected"), REPO)


def parse(text):
    """(header or None, records) of a file's text."""
    header, records = None, []
    for n, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        o = json.loads(line)
        if n == 0 and "expected" in o and "input" in o:
            header = o
            continue
        records.append(o)
    return header, records


def kind(o):
    ev = o.get("ev", "?")
    if ev == "api":
        return "api:%s" % o.get("name")
    if ev == "outcome":
        return "outcome:%s" % o.get("kind")
    return ev


def show(v, limit=60):
    s = json.dumps(v, sort_keys=True)
    return s if len(s) <= limit else s[:limit - 3] + "..."


def field_changes(a, b, path=""):
    """[(sign, path, old, new)]: what changed from record a to record b,
    field by field; lists of records of equal length are compared element by
    element (path "x[]")."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = path + ("." if path else "") + k
            if k not in b:
                out.append(("-", p, a[k], None))
            elif k not in a:
                out.append(("+", p, None, b[k]))
            elif a[k] != b[k]:
                out += field_changes(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a and \
            all(isinstance(x, dict) for x in a + b):
        for x, y in zip(a, b):
            if x != y:
                out += field_changes(x, y, path + "[]")
        return out
    return [("~", path, a, b)]


class Summary:
    def __init__(self):
        self.kinds = collections.defaultdict(collections.Counter)  # kind -> {added, removed, changed}
        self.fields = collections.defaultdict(collections.Counter)  # kind -> {(sign, path): n}
        self.values = collections.defaultdict(collections.Counter)  # (kind, path) -> {(old, new): n}
        self.scenarios = collections.defaultdict(lambda: {"files": 0, "records": 0, "kinds": set()})
        self.files = collections.Counter()

    def pair(self, rel, old, new):
        """Adds the change from old to new (each a text, or None when the
        file is absent) of the file rel (<module>/<scenario>/<setup>.jsonl)."""
        parts = rel.split(os.sep)
        scen = "/".join(parts[:2]) if len(parts) >= 3 else rel
        if old is None or new is None:
            self.files["added" if old is None else "removed"] += 1
            s = self.scenarios[scen]
            s["files"] += 1
            s["kinds"].add("(file %s)" % ("added" if old is None else "removed"))
            return
        oh, orecs = parse(old)
        nh, nrecs = parse(new)
        if orecs == nrecs:
            if oh != nh:
                self.files["header only"] += 1
            return
        self.files["changed"] += 1
        s = self.scenarios[scen]
        s["files"] += 1
        a = [json.dumps(o, sort_keys=True) for o in orecs]
        b = [json.dumps(o, sort_keys=True) for o in nrecs]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            olds, news = orecs[i1:i2], nrecs[j1:j2]
            # Pair each removed record with the next added one of its kind,
            # in order; the rest were added or removed.
            used = set()
            for o in olds:
                k = kind(o)
                match = next((n for n, r in enumerate(news) if n not in used and kind(r) == k), None)
                s["records"] += 1
                s["kinds"].add(k)
                if match is None:
                    self.kinds[k]["removed"] += 1
                    continue
                used.add(match)
                self.kinds[k]["changed"] += 1
                for sign, path, x, y in field_changes(o, news[match]):
                    self.fields[k][(sign, path)] += 1
                    if sign == "~":
                        self.values[(k, path)][(show(x), show(y))] += 1
            for n, r in enumerate(news):
                if n not in used:
                    k = kind(r)
                    self.kinds[k]["added"] += 1
                    s["records"] += 1
                    s["kinds"].add(k)

    def report(self, samples=3, scenarios=40):
        lines = []
        f = self.files
        lines.append("files: %d changed, %d added, %d removed, %d with only the header changed" % (
            f["changed"], f["added"], f["removed"], f["header only"]))
        lines.append("")
        lines.append("by record kind:")
        for k in sorted(self.kinds, key=lambda k: (-sum(self.kinds[k].values()), k)):
            c = self.kinds[k]
            lines.append("  %-28s %s" % (k, ", ".join("%d %s" % (c[x], x) for x in ("changed", "added", "removed") if c[x])))
            for (sign, path), n in sorted(self.fields[k].items(), key=lambda x: (-x[1], x[0])):
                lines.append("      %s%-32s %d" % (sign, path, n))
                if sign == "~":
                    for (x, y), m in self.values[(k, path)].most_common(samples):
                        lines.append("          %s -> %s  (%d)" % (x, y, m))
        lines.append("")
        lines.append("by scenario (%d):" % len(self.scenarios))
        ranked = sorted(self.scenarios.items(), key=lambda x: (-x[1]["records"], x[0]))
        for name, s in ranked[:scenarios]:
            lines.append("  %-60s %d files, %d records: %s" % (name, s["files"], s["records"], ", ".join(sorted(s["kinds"]))))
        if len(ranked) > scenarios:
            lines.append("  ... and %d more (--scenarios N)" % (len(ranked) - scenarios))
        return "\n".join(lines)


def _git(*args):
    return subprocess.run(["git"] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def git_pairs(a, b):
    """(rel, old text, new text) for each expected file that differs between
    revision a and revision b (None: the working tree)."""
    revs = [a] + ([b] if b else [])
    out = _git("diff", "--no-renames", "--name-status", *revs, "--", EXPECTED_REL)
    for line in out.splitlines():
        status, path = line.split("\t", 1)
        rel = os.path.relpath(path, EXPECTED_REL)

        def at(rev):
            if rev is None:
                p = os.path.join(REPO, path)
                return open(p).read() if os.path.exists(p) else None
            try:
                return _git("show", "%s:%s" % (rev, path))
            except subprocess.CalledProcessError:
                return None
        yield rel, (None if status.startswith("A") else at(a)), (None if status.startswith("D") else at(b))


def dir_pairs(old, new):
    """(rel, old text, new text) for each file of either tree."""
    def files(root):
        got = {}
        for d, _, names in os.walk(root):
            for n in names:
                if n.endswith(".jsonl"):
                    p = os.path.join(d, n)
                    got[os.path.relpath(p, root)] = p
        return got
    a, b = files(old), files(new)
    for rel in sorted(set(a) | set(b)):
        x = open(a[rel]).read() if rel in a else None
        y = open(b[rel]).read() if rel in b else None
        if x != y:
            yield rel, x, y


def cmd_diffsum(args):
    if args.dirs:
        pairs = dir_pairs(*args.dirs)
    else:
        revs = args.revs or ["HEAD"]
        if len(revs) > 2:
            raise SystemExit("diffsum: at most two revisions")
        pairs = git_pairs(revs[0], revs[1] if len(revs) > 1 else None)
    pats = args.k.split(",") if args.k else None
    s = Summary()
    for rel, old, new in pairs:
        parts = rel.split(os.sep)
        if pats and not (len(parts) >= 2 and matches(parts[1], parts[0], pats)):
            continue
        s.pair(rel, old, new)
    print(s.report(args.samples, args.scenarios))
    return 0

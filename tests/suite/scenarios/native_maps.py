"""Memory maps of 24 to 64 ranges (pgtest/maps_util.py): the maps generator's
programs at fixed seeds, chosen for how their descriptors list ranges that
share a start.

The memory map sorts its ranges by start and then length, so zero-length
ranges come before the range with a length that shares their start, and
refuses the map when a range starts before the previous range's end. A map
whose only shared starts are zero-length ranges beside at most one range
with a length is therefore accepted, whatever order the host listed them in:
- `tie_length_first_*`: the descriptor lists the range with a length before
  the zero-length ranges sharing its start;
- `tie_zero_first_*`: the other way round;
- `both_accept_*`: zero-length ranges already listed first where they share
  a start with a range with a length (24), or only at starts no range with a
  length has (64);
- `no_zero_*`: every range has a length (24, 52 and 64 of them), adjacent
  or apart.
Every one of these maps makes a device. A task's buffers then lie over the
ranges: across adjacent ranges (mapped a piece a range), and with a page in a
gap between them or past them (the exec faults). Each map's descriptor order,
and that the device is made, are predicted by maps_util, and each scenario
checks its prediction when it is built.

Setups: map and isolate (the blocks setup has no memory map). Under
isolation a map with a zero-length range makes no isolated task, since
every range must be whole pages (docs/testing.md, section 3.3): the define
faults, and the waits for the task's work fail at their timeouts. Where
those add up past the run's 30 s watchdog (both_accept_64,
tie_length_first_64, tie_zero_first_24), the watchdog ends the run.
"""
from pgtest.maps_util import build
from pgtest.registry import scenario

# (name, seed, ranges, how the descriptor lists a shared start's ranges)
CASES = [
    ("tie_length_first_24", 7, 24, "nonzero_first"),
    ("tie_length_first_52", 24, 52, "nonzero_first"),
    ("tie_length_first_64", 40, 64, "nonzero_first"),
    ("tie_zero_first_24", 1, 24, "nonzero_last"),
    ("tie_zero_first_64", 3, 64, "nonzero_last"),
    ("both_accept_24", 2, 24, "nonzero_last"),
    ("both_accept_64", 8, 64, "shuffled"),
    ("no_zero_24", 9, 24, "shuffled"),
    ("no_zero_52", 10, 52, "shuffled"),
    ("no_zero_64", 10, 64, "nonzero_first"),
]


@scenario(setups=("map", "isolate"))
def native_maps():
    out = []
    for name, seed, n, mode in CASES:
        s, plan = build(seed, n, __doc__)
        assert len(plan.order) == n, name
        assert plan.ok, "%s: %s" % (name, plan.params())
        assert plan.mode == mode, "%s: %s" % (name, plan.params())
        assert plan.mixed_runs > 0 or not name.startswith("tie_"), "%s: %s" % (name, plan.params())
        assert (plan.zeros == 0) == name.startswith("no_zero"), "%s: %s" % (name, plan.params())
        s.name = "native_maps_" + name
        out.append(s)
    return out

"""Tests of the builders behind the native_maps, native_alias and
native_teardown scenarios: maps_util's statement of the memory map's rule,
and that every builder makes its programs, deterministically. No GPU, no
runner.

  cd tests/suite && python3 -m unittest pgtest.test_reach -v
"""
import json
import unittest

from . import abi, alias_util, maps_util, teardown_util


class MapRule(unittest.TestCase):
    def test_accepts_zero_length_ranges_in_either_order(self):
        a = maps_util.accepts
        self.assertTrue(a([(0x1000, 0x1000, 0), (0x1000, 0, 1)]))
        self.assertTrue(a([(0x1000, 0, 1), (0x1000, 0x1000, 0)]))
        self.assertTrue(a([(0x2000, 0, 1), (0x1000, 0x1000, 0), (0x2000, 0x1000, 2), (0x2000, 0, 3)]))
        self.assertFalse(a([(0x1000, 0x2000, 0), (0x1000, 0x1000, 1)]))
        self.assertFalse(a([(0x1000, 0x2000, 0), (0x2000, 0, 1)]))

    def test_the_sort_is_by_start_then_length_and_stable(self):
        v = [(0x2000, 0x1000, "a"), (0x1000, 0, "b"), (0x2000, 0, "c"), (0x1000, 0, "d")]
        self.assertEqual([x[2] for x in sorted(v, key=maps_util.sort_key)], ["b", "d", "c", "a"])

    def test_accepted(self):
        a = maps_util.accepted
        self.assertTrue(a([(0x1000, 0, 0), (0x1000, 0x1000, 0), (0x2000, 0x1000, 0)]))
        self.assertFalse(a([(0x1000, 0x1000, 0), (0x1000, 0, 0)]))
        self.assertTrue(a([(0x1000, 0x1000, 0), (0x2000, 0, 0), (0x2000, 0, 0)]))
        self.assertFalse(a([(0x1000, 0x2000, 0), (0x2000, 0x1000, 0)]))


def maps(seed, n):
    s, plan = maps_util.build(seed, n, "doc")
    return s, plan.params()


def alias(seed, n):
    return alias_util.build(seed, n, "doc", intra=False)


def alias_intra(seed, n):
    return alias_util.build(seed, n, "doc", intra=True)


def teardown(seed, n):
    return teardown_util.build(seed, n, "doc")


class Generators(unittest.TestCase):
    def test_programs_build_and_repeat(self):
        for build, sizes in ((maps, range(24, 65)), (alias, range(2, 25)), (alias_intra, range(2, 25)),
                             (teardown, range(1, 13))):
            for seed in range(40):
                n = sizes[seed % len(sizes)]
                s, params = build(seed, n)
                again, params_again = build(seed, n)
                key = "%s:%d:%d" % (build.__name__, seed, n)
                self.assertEqual(s.jsonl(), again.jsonl(), key)
                self.assertEqual(params, params_again, key)

    def test_alias_intra_is_asked_for(self):
        self.assertTrue(all("intra=no" in alias(s, 2 + s % 23)[1] for s in range(20)))
        self.assertTrue(all("intra=yes" in alias_intra(s, 2 + s % 23)[1] for s in range(20)))

    def test_maps_reach(self):
        """24-64 ranges; runs of equal starts listed with the range with a
        length first and last, and every map accepted."""
        kinds = set()
        for seed in range(200):
            n = 24 + seed % 41
            s, params = maps(seed, n)
            ranges = json.loads(s.jsonl().splitlines()[1])["descriptor"]["map_ranges"]
            self.assertEqual(len(ranges), n, (seed, n))
            self.assertIn("map=ok", params.split(), (seed, n))
            fields = dict(x.split("=", 1) for x in params.split())
            if fields["mixed"] != "0":
                kinds.add(fields["order"])
        self.assertIn("nonzero_first", kinds)
        self.assertIn("nonzero_last", kinds)

    def test_teardown_efi_presents_nothing(self):
        """A teardown program's EFI frame has height 0, so its timer's ticks
        present nothing and call no new_frame back: a new_frame between the
        last checkpoint and the release was timing under load (1 of 24 EFI
        release programs in a 150-program random run, 1 in 90 runs of
        native_teardown_2_5[isolate]). Its other registers stay valid, so
        the timer runs up to the teardown."""
        efi = 0
        for seed in range(300):
            n = 1 + seed % 12
            s, params = teardown(seed, n)
            key = "%d:%d" % (seed, n)
            w = {}
            for o in s.ops:
                if o.get("op") == "mmio_w":
                    w.setdefault(int(o["off"], 16), []).append(int(o["val"], 16))
            if "EFI timer" not in params:
                self.assertNotIn(abi.R_EFI_ENABLE, w, key)
                continue
            efi += 1
            self.assertEqual(w[abi.R_EFI_ENABLE], [1], key)
            mode = w[abi.R_EFI_MODE]
            self.assertEqual(len(mode), 1, key)
            width, height = mode[0] & 0xFFFF, mode[0] >> 16
            self.assertEqual(height, 0, key)
            self.assertGreater(width, 0, key)
            self.assertEqual(w[abi.R_EFI_DEPTH], [32], key)
            stride = w[abi.R_EFI_STRIDE][0]
            self.assertTrue(stride >= 4 * width and stride % 16 == 0, key)
            for o in s.ops:
                self.assertFalse(o.get("op") == "wait" and o.get("cb") in ("modeChange", "newFrame"), key)
                self.assertNotIn("collapse", o, key)
        self.assertGreater(efi, 40)


if __name__ == "__main__":
    unittest.main()

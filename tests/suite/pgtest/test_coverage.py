"""Tests of the coverage tooling's pure parts (pgtest/coverage.py): the
build environment it derives, the line union and the report. They run no
build, no case and no GPU.

  cd tests/suite && python3 -m unittest pgtest.test_coverage -v
"""
import os
import unittest

from . import cli
from . import coverage


class BuildEnv(unittest.TestCase):
    def test_show_env_exports_are_read(self):
        text = ("export LLVM_PROFILE_FILE='/t/x-%p-%16m.profraw'\n"
                "export RUSTC_WRAPPER=/h/.cargo/bin/cargo-llvm-cov\n"
                "export __CARGO_LLVM_COV_RUSTC_WRAPPER_RUSTFLAGS='-Cinstrument-coverage'\n"
                "export CARGO_LLVM_COV_SHOW_ENV=1\n"
                "not an export\n")
        env = coverage.parse_show_env(text)
        self.assertEqual(env["LLVM_PROFILE_FILE"], "/t/x-%p-%16m.profraw")
        self.assertEqual(env["RUSTC_WRAPPER"], "/h/.cargo/bin/cargo-llvm-cov")
        self.assertEqual(env["__CARGO_LLVM_COV_RUSTC_WRAPPER_RUSTFLAGS"], "-Cinstrument-coverage")
        self.assertEqual(len(env), 4)

    def test_ccc_override_appends_the_runtime_and_page_aligned_counters(self):
        edits = coverage.ccc_override("/rt/libprofiler_builtins-x.rlib").split(" ")
        # Through -Wl, so that a compile under -xobjective-c never takes it for a source.
        self.assertEqual(edits[0], "+-Wl,/rt/libprofiler_builtins-x.rlib")
        self.assertEqual(sorted(edits[1:]), sorted(
            "+-Wl,-sectalign,__DATA,%s,0x4000" % s
            for s in ("__llvm_prf_cnts", "__llvm_prf_bits", "__llvm_prf_data")))

    def test_the_build_is_a_dist_tree(self):
        self.assertTrue(coverage.COV_DIST.endswith("/build/cov/dist"))
        self.assertEqual(cli.LIB_BIN, "lib/libreims_vgpu.dylib")
        self.assertEqual(coverage.objects("/D"), {"host": "/D/" + cli.LIB_BIN})


class Reports(unittest.TestCase):
    def test_rel_keeps_files_in_the_repository_only(self):
        self.assertEqual(coverage.rel(os.path.join(cli.REPO, "crates/a/src/x.rs")), "crates/a/src/x.rs")
        self.assertIsNone(coverage.rel("/rustc/abc/library/core/src/lib.rs"))
        self.assertEqual(coverage.rel("crates/a/src/x.rs"), "crates/a/src/x.rs")

    def test_report_counts_lines_per_file_crate_and_total(self):
        lines = {
            "crates/reims-vgpu-core/src/a.rs": {1: 3, 2: 0, 3: 1},
            "crates/reims-vgpu-core/src/b.rs": {5: 0},
            "crates/reims-vgpu-c/src/device.rs": {1: 1, 2: 1},
        }
        text = coverage.report_text(lines)
        rows = {l.split()[0]: l.split()[1:] for l in text.splitlines() if l.strip()}
        self.assertEqual(rows["crates/reims-vgpu-core/src/a.rs"], ["2", "3", "66.67%"])
        self.assertEqual(rows["crates/reims-vgpu-core"], ["2", "4", "50.00%"])
        self.assertEqual(rows["crates/reims-vgpu-c"], ["2", "2", "100.00%"])
        self.assertEqual(rows["TOTAL"], ["4", "6", "66.67%"])
        # A pattern selects the files shown; the totals stay whole.
        text = coverage.report_text(lines, "b.rs")
        files = [l.split()[0] for l in text.splitlines() if l.startswith("crates/") and ".rs" in l]
        self.assertEqual(files, ["crates/reims-vgpu-core/src/b.rs"])
        self.assertIn("TOTAL", text)

    def test_safe_case_ids(self):
        self.assertEqual(coverage.safe("fifo_nop[map-isolate]"), "fifo_nop_map-isolate_")


if __name__ == "__main__":
    unittest.main()

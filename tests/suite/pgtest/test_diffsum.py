"""Unit tests of diffsum's pure parts: no git, no runner."""
import json
import unittest

from . import diffsum


def text(records, header=True):
    lines = [json.dumps({"expected": "x", "input": "h", "setup": "map", "host": None})] if header else []
    return "\n".join(lines + [json.dumps(r) for r in records]) + "\n"


class FieldChanges(unittest.TestCase):
    def test_added_removed_and_changed_fields(self):
        got = diffsum.field_changes({"a": 1, "b": 2}, {"b": 3, "c": 4})
        self.assertEqual(got, [("-", "a", 1, None), ("~", "b", 2, 3), ("+", "c", None, 4)])

    def test_lists_of_records_compare_element_by_element(self):
        a = {"callbacks": [{"cb": "x", "bpr": 4}, {"cb": "y"}]}
        b = {"callbacks": [{"cb": "x"}, {"cb": "y"}]}
        self.assertEqual(diffsum.field_changes(a, b), [("-", "callbacks[].bpr", 4, None)])

    def test_lists_of_other_lengths_change_whole(self):
        got = diffsum.field_changes({"r": [1]}, {"r": [1, 2]})
        self.assertEqual(got, [("~", "r", [1], [1, 2])])


class Summary(unittest.TestCase):
    def test_counts_by_kind_and_scenario(self):
        s = diffsum.Summary()
        old = [{"ev": "binding", "lib": "a"}, {"ev": "encode", "status": 4}, {"ev": "outcome", "kind": "completed"}]
        new = [{"ev": "binding", "lib": "b"}, {"ev": "encode", "status": "RVG_OK"}, {"ev": "mark"},
               {"ev": "outcome", "kind": "completed"}]
        s.pair("m/scen/map.jsonl", text(old), text(new))
        self.assertEqual(s.kinds["binding"]["changed"], 1)
        self.assertEqual(s.kinds["encode"]["changed"], 1)
        self.assertEqual(s.kinds["mark"]["added"], 1)
        self.assertEqual(s.values[("encode", "status")][("4", '"RVG_OK"')], 1)
        self.assertEqual(s.scenarios["m/scen"]["files"], 1)
        self.assertIn("encode", s.report())

    def test_a_header_alone_and_whole_files(self):
        s = diffsum.Summary()
        recs = [{"ev": "end"}]
        s.pair("m/a/map.jsonl", text(recs), text(recs).replace('"h"', '"g"'))
        s.pair("m/b/map.jsonl", None, text(recs))
        s.pair("m/c/map.jsonl", text(recs), None)
        self.assertEqual(s.files, {"header only": 1, "added": 1, "removed": 1})

    def test_api_and_outcome_kinds(self):
        self.assertEqual(diffsum.kind({"ev": "api", "name": "reset"}), "api:reset")
        self.assertEqual(diffsum.kind({"ev": "outcome", "kind": "died"}), "outcome:died")
        self.assertEqual(diffsum.parse(text([{"ev": "end"}], header=False)), (None, [{"ev": "end"}]))


if __name__ == "__main__":
    unittest.main()

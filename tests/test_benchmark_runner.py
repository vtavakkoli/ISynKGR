from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from isynkgr.evaluation.benchmark_runner import build_pairs, load_standards, save_outputs


class BenchmarkRunnerTests(unittest.TestCase):
    def test_build_pairs_excludes_identity_pairs(self) -> None:
        standards = ["ieee1451", "iso15926", "iec61499"]
        pairs = build_pairs(standards)
        self.assertEqual(len(pairs), 6)
        self.assertNotIn(("ieee1451", "ieee1451"), pairs)

    def test_load_standards_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "standards.json"
            cfg.write_text(json.dumps({"standards": ["ieee1451", "iso15926"]}))
            self.assertEqual(load_standards(cfg), ["ieee1451", "iso15926"])

    def test_save_outputs_replaces_latest_symlink(self) -> None:
        row = {
            "source": "ieee1451",
            "target": "iso15926",
            "method": "isynkgr",
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "property_accuracy": 1.0,
            "latency_s_avg": 0.01,
            "latency_s_total": 0.01,
            "token_count_avg": 0,
            "kg_traversed_edges_avg": 1,
            "cpu_time_s": 0.01,
            "peak_rss_mb": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            old.mkdir()
            (root / "latest").symlink_to(old.name)
            out = save_outputs([row], root)
            self.assertTrue((root / "latest").is_dir())
            self.assertTrue((out / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()

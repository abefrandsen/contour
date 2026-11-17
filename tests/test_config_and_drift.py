from pathlib import Path
import tempfile
import unittest

from contour.bundle import SCHEMA_ID, validate_bundle
from contour.config import CONFIG_FILE, ConfigError, find_root, load_config
from contour.drift import find_drift_issues
from contour.graph import ContourGraph
from contour.ingest import ingest_bundle
from contour.mdio import parse_markdown_file, write_markdown_file


class TestConfigAndDrift(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / CONFIG_FILE).write_text("contour: 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_find_root_from_nested_dir(self) -> None:
        nested = self.root / "a" / "b"
        nested.mkdir(parents=True)
        self.assertEqual(find_root(start=nested), self.root.resolve())

    def test_invalid_config_raises(self) -> None:
        (self.root / CONFIG_FILE).write_text("contour: 99\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_drift_detects_missing_bad_validity_and_unlinked(self) -> None:
        b = validate_bundle({
            "schema": SCHEMA_ID,
            "run_id": "run_bad_1",
            "status": "final",
            "created_at": "2026-11-16T10:00:00Z",
            "finished_at": "2026-11-16T11:00:00Z",
            "facets": {"task": "cls", "arch_family": "vit", "arch_scale": "22M"},
            "data": {"train": ["tr"], "eval": ["ev"]},
            "lineage": {"motivated_by": ["crn_a"]},
            "metrics": {"ev/top1": 70.0},
        })
        res = ingest_bundle(self.root, b)
        # Mark fieldlog as bugged and cite it anyway
        fm, body = parse_markdown_file(res.path)
        fm["validity"] = "bugged"
        write_markdown_file(res.path, fm, body)

        write_markdown_file(
            self.root / "cairns" / "crn_a.md",
            {
                "id": "crn_a",
                "type": "cairn",
                "title": "A",
                "status": "active",
                "needs_synthesis": True,
                "supports": [f"[[{res.log_id}]]", "[[log_missing]]"],
            },
            "Body\n",
        )
        g = ContourGraph(self.root)
        codes = {i.code for i in find_drift_issues(g)}
        self.assertIn("H001", codes)  # needs_synthesis
        self.assertIn("H002", codes)  # bugged evidence cited
        self.assertIn("H004", codes)  # missing fieldlog


if __name__ == "__main__":
    unittest.main()

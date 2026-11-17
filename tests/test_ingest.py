from pathlib import Path
import tempfile
import unittest

from contour.bundle import SCHEMA_ID, validate_bundle
from contour.ingest import FieldlogIndex, ingest_bundle, relink_parents
from contour.mdio import parse_markdown_file, write_markdown_file


def _bundle(run_id: str, status: str = "running", **kw):
    b = {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "title": f"Run {run_id}",
        "status": status,
        "created_at": "2026-11-16T10:00:00Z",
        "facets": {"task": "cls", "arch_family": "vit", "arch_scale": "22M"},
        "data": {"train": ["tr"], "eval": ["ev"]},
        "metrics": {"ev/top1": 75.0} if status == "final" else {},
    }
    if status in ("final", "failed"):
        b["finished_at"] = "2026-11-16T11:00:00Z"
    b.update(kw)
    return validate_bundle(b)


class TestIngest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "cairns").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_preserves_human_fields_and_blocks_terminal_overwrite(self) -> None:
        write_markdown_file(
            self.root / "cairns" / "crn_test.md",
            {"id": "crn_test", "type": "cairn", "title": "T", "status": "speculative", "needs_synthesis": False},
            "## Proposition\n- Test\n",
        )
        idx = FieldlogIndex(self.root)
        r1 = ingest_bundle(self.root, _bundle("run_0001", "running", lineage={"motivated_by": ["crn_test"]}), idx)
        self.assertEqual(r1.action, "created")
        self.assertEqual(r1.flagged, [])

        # Human edits validity, title, and body
        fm, _ = parse_markdown_file(r1.path)
        fm["validity"] = "bugged"
        fm["title"] = "Custom Human Title"
        write_markdown_file(r1.path, fm, "## Key Takeaway (TL;DR)\n- Bugged LR.\n")

        # Update to final preserves human edits and flags cairn
        r2 = ingest_bundle(self.root, _bundle("run_0001", "final", lineage={"motivated_by": ["crn_test"]}), idx)
        self.assertEqual(r2.action, "updated")
        self.assertEqual(len(r2.flagged), 1)
        fm2, body2 = parse_markdown_file(r1.path)
        self.assertEqual(fm2["validity"], "bugged")
        self.assertEqual(fm2["title"], "Custom Human Title")
        self.assertIn("Bugged LR.", body2)
        cfm, _ = parse_markdown_file(self.root / "cairns" / "crn_test.md")
        self.assertTrue(cfm["needs_synthesis"])

        # Re-ingesting identical final bundle is unchanged
        r3 = ingest_bundle(self.root, _bundle("run_0001", "final", lineage={"motivated_by": ["crn_test"]}), idx)
        self.assertEqual(r3.action, "unchanged")

        # Mutating terminal bundle is rejected without force
        r4 = ingest_bundle(self.root, _bundle("run_0001", "final", metrics={"ev/top1": 99.0}), idx)
        self.assertEqual(r4.action, "rejected")

        # With force=True it succeeds
        r5 = ingest_bundle(self.root, _bundle("run_0001", "final", metrics={"ev/top1": 99.0}), idx, force=True)
        self.assertEqual(r5.action, "updated")

    def test_relink_out_of_order_parents(self) -> None:
        idx = FieldlogIndex(self.root)
        rc = ingest_bundle(self.root, _bundle("child_run_1", "final", lineage={"parents": ["parent_run_1"]}), idx)
        fm_before, _ = parse_markdown_file(rc.path)
        self.assertEqual(fm_before["parent_runs"], ["run:parent_run_1"])

        rp = ingest_bundle(self.root, _bundle("parent_run_1", "final"), idx)
        relinked = relink_parents(self.root, idx)
        self.assertEqual(relinked, [rc.log_id])
        fm_after, _ = parse_markdown_file(rc.path)
        self.assertEqual(fm_after["parent_runs"], [f"[[{rp.log_id}]]"])


if __name__ == "__main__":
    unittest.main()

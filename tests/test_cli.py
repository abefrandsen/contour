import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from contour.bundle import SCHEMA_ID
from contour.cli import main
from contour.mdio import parse_markdown_file


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        main(["init", str(self.root)])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_json(self, argv: list[str]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--root", str(self.root), "--json", *argv])
        return json.loads(buf.getvalue())

    def test_end_to_end_sync_ridge_bearings_new_link(self) -> None:
        inbox = self.root / "field_bundles"
        bundles = [
            {
                "schema": SCHEMA_ID,
                "run_id": "base_0001",
                "title": "Baseline",
                "status": "final",
                "created_at": "2026-11-16T10:00:00Z",
                "finished_at": "2026-11-16T11:00:00Z",
                "facets": {"task": "lm", "arch_family": "gpt", "arch_scale": "125M"},
                "data": {"train": ["pile"], "eval": ["wikitext"]},
                "compute": {"gpu_hours": 10.0},
                "metrics": {"wikitext/ppl": 20.0},
            },
            {
                "schema": SCHEMA_ID,
                "run_id": "child_a_01",
                "title": "RoPE",
                "status": "final",
                "created_at": "2026-11-16T12:00:00Z",
                "finished_at": "2026-11-16T13:00:00Z",
                "facets": {"task": "lm", "arch_family": "gpt", "arch_scale": "125M"},
                "data": {"train": ["pile"], "eval": ["wikitext"]},
                "interventions": ["rope"],
                "lineage": {"parents": ["base_0001"]},
                "compute": {"gpu_hours": 10.0},
                "metrics": {"wikitext/ppl": 18.5},
            },
            {
                "schema": SCHEMA_ID,
                "run_id": "child_b_01",
                "title": "SwiGLU + RMSNorm",
                "status": "final",
                "created_at": "2026-11-16T14:00:00Z",
                "finished_at": "2026-11-16T15:00:00Z",
                "facets": {"task": "lm", "arch_family": "gpt", "arch_scale": "125M"},
                "data": {"train": ["pile"], "eval": ["wikitext"]},
                "interventions": ["swiglu", "rmsnorm"],
                "lineage": {"parents": ["base_0001"]},
                "compute": {"gpu_hours": 9.5},
                "metrics": {"wikitext/ppl": 18.0},
            },
        ]
        for i, b in enumerate(bundles):
            (inbox / f"b{i}.json").write_text(json.dumps(b))

        sync_out = self._run_json(["sync"])
        self.assertEqual(len(sync_out["results"]), 3)
        self.assertEqual(sync_out["errors"], [])

        # Child fieldlog has embedded vs_parent computed automatically
        child_a_id = sync_out["results"][1]["log_id"]
        insp = self._run_json(["inspect", child_a_id])
        vs = insp["frontmatter"]["vs_parent"]
        self.assertIsNotNone(vs)
        self.assertTrue(vs["single_logical_diff"])
        self.assertEqual(vs["added_interventions"], ["rope"])
        self.assertAlmostEqual(vs["deltas"]["wikitext/ppl"]["delta"], -1.5)

        ridge = self._run_json(["ridge", "--eval", "wikitext"])
        self.assertEqual(ridge["direction"], "min")
        self.assertEqual(ridge["runs"][0]["value"], 18.0)

        bearings = self._run_json(["bearings"])
        self.assertEqual(len(bearings["move_3_synthesis"]), 1)
        self.assertEqual(
            bearings["move_3_synthesis"][0]["combined_interventions"],
            ["rmsnorm", "rope", "swiglu"],
        )
        self.assertEqual(len(bearings["move_4_occam"]), 1)

        # `new` safely serializes quotes in title and `link` attaches evidence
        main(["--root", str(self.root), "new", "cairn", "qtest", "--title", 'Claim with "quotes"'])
        link_out = self._run_json(["link", child_a_id, "crn_qtest", "--relation", "supports"])
        self.assertEqual(link_out["relation"], "supports")
        fm, _ = parse_markdown_file(self.root / "cairns" / "crn_qtest.md")
        self.assertEqual(fm["title"], 'Claim with "quotes"')
        self.assertEqual(fm["supports"], [f"[[{child_a_id}]]"])

    def test_init_subdir_with_skill_and_ejected_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                main([
                    "--json",
                    "init",
                    str(repo / "contour"),
                    "--with-skill",
                    "--eject-logger",
                    str(repo / "contour_run.py"),
                ])
            out = json.loads(buf.getvalue())
            self.assertTrue((repo / "contour" / "contour.yaml").exists())
            self.assertTrue((repo / ".agent" / "skills" / "contour-research" / "SKILL.md").exists())
            self.assertTrue((repo / "contour_run.py").exists())
            self.assertIn("root", out)


if __name__ == "__main__":
    unittest.main()

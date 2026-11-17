import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from contour.bundle import SCHEMA_ID, validate_bundle
from contour.cli import main
from contour.drift import find_drift_issues
from contour.graph import ContourGraph
from contour.ingest import FieldlogIndex, ingest_bundle
from contour.mdio import parse_markdown_file


class TestCompareAndHaze(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        main(["init", str(self.root)])
        self.idx = FieldlogIndex(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _add_run(
        self,
        run_id: str,
        top1: float,
        *,
        seed: int = 0,
        train: list[str] | None = None,
        interventions: list[str] | None = None,
        parents: list[str] | None = None,
        tier: str = "proxy",
        params: int | None = None,
    ) -> str:
        b = validate_bundle({
            "schema": SCHEMA_ID,
            "run_id": run_id,
            "title": run_id,
            "status": "final",
            "created_at": "2026-11-16T10:00:00Z",
            "finished_at": "2026-11-16T11:00:00Z",
            "facets": {
                "task": "cls",
                "arch_family": "vit",
                "arch_scale": "22M",
                "tier": tier,
                "seed": seed,
            },
            "data": {"train": train or ["ds_tr"], "eval": ["ds_ev"]},
            "interventions": interventions or [],
            "lineage": {"parents": parents or []},
            "compute": {"gpu_hours": 10.0, **({"params": params} if params else {})},
            "metrics": {"ds_ev/top1": top1},
        })
        return ingest_bundle(self.root, b, index=self.idx).log_id

    def _run_json(self, argv: list[str]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--root", str(self.root), "--json", *argv])
        return json.loads(buf.getvalue())

    def test_embedded_vs_parent_and_haze_refresh(self) -> None:
        b0 = self._add_run("base_seed0", 79.6, seed=0)
        b1 = self._add_run("base_seed1", 79.8, seed=1)
        b2 = self._add_run("base_seed2", 79.7, seed=2)
        t_big = self._add_run("treat_big", 80.3, seed=0, interventions=["qk_rmsnorm"], parents=[b0])
        t_small = self._add_run("treat_small", 79.75, seed=0, interventions=["tiny_tweak"], parents=[b0])

        # Before haze: vs_parent is Grade B
        fm_pre, _ = parse_markdown_file(self.root / "fieldlogs" / f"{t_big}.md")
        self.assertEqual(fm_pre["vs_parent"]["grade"], "B")

        # Seed noise floor from 3 baseline seeds (std = 0.1) -> automatically refreshes vs_parent on fieldlogs
        noise_out = self._run_json(["noise", "seed", "cls/ds_ev/22M", "--from", b0, b1, b2])
        self.assertAlmostEqual(noise_out["entries"]["ds_ev/top1"]["std"], 0.1, places=4)

        fm_big, _ = parse_markdown_file(self.root / "fieldlogs" / f"{t_big}.md")
        self.assertEqual(fm_big["vs_parent"]["grade"], "A")
        self.assertTrue(fm_big["vs_parent"]["deltas"]["ds_ev/top1"]["clears_noise"])

        fm_small, _ = parse_markdown_file(self.root / "fieldlogs" / f"{t_small}.md")
        self.assertEqual(fm_small["vs_parent"]["grade"], "B")
        self.assertFalse(fm_small["vs_parent"]["deltas"]["ds_ev/top1"]["clears_noise"])

        # Citing t_small under supports triggers H011 (within haze noise floor)
        main(["--root", str(self.root), "new", "cairn", "tiny", "--title", "Tiny tweak helps"])
        main(["--root", str(self.root), "link", t_small, "crn_tiny", "--relation", "supports"])
        codes = {i.code for i in find_drift_issues(ContourGraph(self.root))}
        self.assertIn("H011", codes)

    def test_confounds_flagged_in_drift_and_bearings(self) -> None:
        b0 = self._add_run("base_01", 79.7, params=22_000_000)
        t_data = self._add_run(
            "treat_data", 81.0, train=["ds_tr", "extra_ds"], interventions=["lr_2x"], parents=[b0]
        )

        cmp_data = self._run_json(["compare", b0, "--vs", t_data])
        self.assertEqual(cmp_data["grade"], "C")
        self.assertFalse(cmp_data["confounds"]["single_logical_diff"])
        self.assertIn("dataset_train", cmp_data["facet_diff"]["confounding_facets"])

        # No sightings/ directory created
        self.assertFalse((self.root / "sightings").exists())

        codes = {i.code for i in find_drift_issues(ContourGraph(self.root))}
        self.assertIn("H012", codes)

        bearings = self._run_json(["bearings"])
        self.assertEqual(len(bearings["move_1_confound_isolation"]), 1)
        self.assertEqual(bearings["move_1_confound_isolation"][0]["run"], t_data)

    def test_multiseed_on_the_fly_compare(self) -> None:
        c0 = self._add_run("ctrl_s0", 79.6, seed=0)
        c1 = self._add_run("ctrl_s1", 79.8, seed=1)
        t0 = self._add_run("trt_s0", 80.4, seed=0, interventions=["qk_rmsnorm"])
        t1 = self._add_run("trt_s1", 80.6, seed=1, interventions=["qk_rmsnorm"])

        cmp_out = self._run_json(["compare", c0, c1, "--vs", t0, t1])
        self.assertEqual(cmp_out["grade"], "A")
        self.assertIsNotNone(cmp_out["metrics"]["ds_ev/top1"]["ci_95"])


if __name__ == "__main__":
    unittest.main()

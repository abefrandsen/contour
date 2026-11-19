from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from contour import Analysis, Run
from contour.cli import main
from contour.mdio import parse_markdown_file, write_markdown_file


class TestAnalysesBriefReport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        main(["init", str(self.root)])
        (self.root / "contour.yaml").write_text(yaml.safe_dump({
            "contour": 1,
            "metrics": {
                "val/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
                "ood/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
            },
        }))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--root", str(self.root), "--json", *argv])
        return json.loads(buf.getvalue())

    def test_analyses_ls_brief_and_report_with_asset_export(self) -> None:
        write_markdown_file(
            self.root / "goals" / "goal_enc.md",
            {"title": "Better Vision Encoder", "status": "active"},
            "## Objective & Success Criteria\nImprove accuracy by >=1%.\n\n"
            "## Current Synthesis\n- SigLIP-B looks promising.\n\n"
            "## Open Questions\n"
            "- Does SigLIP beat ResNet on small objects?\n",
        )

        r1 = Run(
            title="ResNet-50 Base",
            config={"encoder": {"type": "resnet50", "depth": 50}, "optim": {"lr": 0.001}},
            root=self.root,
            git_cwd=self.root,
        )
        r1.code = {"commit": "c1", "dirty": False}
        r1.track(79.0, "acc", context={"subset": "val"})
        r1.set_compute_budget(gpu_hours=10.0)
        r1.close()

        r2 = Run(
            title="SigLIP-B Swap",
            parent=r1.id,
            motivated_by=["clm_siglip"],
            config={"encoder": {"type": "siglip_b", "patch": 16}, "optim": {"lr": 0.0005}},
            root=self.root,
            git_cwd=self.root,
        )
        r2.code = {"commit": "c1", "dirty": False}
        r2.track(81.2, "acc", context={"subset": "val"})
        r2.set_compute_budget(gpu_hours=11.0)
        r2.close()

        for r in (r1, r2):
            fm, _ = parse_markdown_file(r.path)
            write_markdown_file(r.path, fm, "## Key Takeaway (TL;DR)\n- Completed cleanly.\n")

        r_live = Run(
            title="SigLIP-L Running",
            artifacts=[{"path": "logs/siglip_l.log", "kind": "log"}],
            root=self.root,
            git_cwd=self.root,
        )

        plot_path = self.root / "analyses" / "artifacts" / "size_curve.png"
        plot_path.write_bytes(b"\x89PNG\r\nfake-plot")
        script_path = self.root / "analyses" / "scripts" / "check_sizes.py"
        script_path.write_text("print('checking sizes')\n")

        # Ad-hoc untracked script -> exploratory -> fine in `qualifies`, does NOT block `report --strict`!
        ana = Analysis(
            title="Object-size error breakdown",
            subjects=[r1.id, r2.id],
            motivated_by=["clm_siglip"],
            script=script_path,
            launch_cmd="python analyses/scripts/check_sizes.py",
            root=self.root,
            git_cwd=self.root,
        )
        ana.code = {"commit": "c1", "dirty": False}
        ana.track(3.1, "small_obj_gain", context={"subset": "diag"})
        ana.add_artifact("analyses/artifacts/size_curve.png", kind="plot", note="Small vs large objects")
        ana.close()
        fm_a, _ = parse_markdown_file(ana.path)
        write_markdown_file(ana.path, fm_a, "## Key Takeaway (TL;DR)\n- Small-object accuracy jumps +3.1%.\n")

        write_markdown_file(
            self.root / "claims" / "clm_siglip.md",
            {
                "title": "SigLIP-B outperforms ResNet-50 on ImageNet-1k classification",
                "status": "supported",
                "verification": "draft",
                "goals": ["[[goal_enc]]"],
                "supports": [],
                "contradicts": [],
                "qualifies": [],
            },
            f"## Hypothesis & Mechanism\nBetter pretraining.\n\n## Evidence Synthesis\n"
            f"- **Holds:** [[{r2.id}]] gains +2.2%; ad-hoc probe [[{ana.id}]] shows small-object gains.\n"
            f"- **Untested / Caveats:** Dense segmentation not yet evaluated.\n",
        )

        # Ad-hoc untracked analysis links under `qualifies` by default
        link_ana_default = self._run_json(["link", ana.id, "clm_siglip"])
        self.assertEqual(link_ana_default["relation"], "qualifies")
        self.assertEqual(link_ana_default["control_level"], "exploratory")
        link_run = self._run_json(["link", r2.id, "clm_siglip"])
        self.assertEqual(link_run["relation"], "supports")

        anas_out = self._run_json(["ls", "analyses", "--kind", "analysis", "--subject", r2.id])
        self.assertEqual(anas_out["count"], 1)
        self.assertEqual(anas_out["analyses"][0]["id"], ana.id)

        show_r2 = self._run_json(["show", r2.id])
        self.assertEqual([a["id"] for a in show_r2["attached_analyses"]], [ana.id])

        brief = self._run_json(["brief", "--goal", "goal_enc"])
        self.assertEqual(len(brief["active_work"]["running_runs"]), 1)
        self.assertEqual(brief["active_work"]["running_runs"][0]["id"], r_live.id)
        self.assertEqual(
            brief["open_threads"]["open_questions"][0]["question"],
            "Does SigLIP beat ResNet on small objects?",
        )

        # `report --strict` fails while claim is `draft`
        with self.assertRaises(SystemExit) as ctx:
            self._run_json(["report", "--goal", "goal_enc", "--strict"])
        self.assertEqual(ctx.exception.code, 1)

        # Verify claim -> `report --strict` passes (exploratory `qualifies` carries no verification weight!)
        main(["--root", str(self.root), "verify", "clm_siglip", "--human", "--note", "Clean swap"])
        export_dir = self.root / "exported_report"
        rep = self._run_json(["report", "--goal", "goal_enc", "--strict", "--out", str(export_dir)])
        self.assertTrue(rep["integrity"]["strict_ok"])
        self.assertEqual(rep["mode"], "verified")
        self.assertTrue((export_dir / "report.md").exists())
        self.assertTrue((export_dir / "assets" / "size_curve.png").exists())
        md = (export_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("assets/size_curve.png", md)
        self.assertIn("Provenance Appendix", md)
        self.assertIn("Dense segmentation not yet evaluated", md)

        prelim = self._run_json(["report", "--runs", r1.id, r2.id])
        self.assertEqual(prelim["mode"], "preliminary")
        self.assertFalse(prelim["integrity"]["strict_ok"])
        self.assertEqual(len(prelim["preliminary_comparisons"]), 1)
        self.assertAlmostEqual(prelim["preliminary_comparisons"][0]["deltas"]["val/acc"]["delta"], 2.2)

        r_live.close()


if __name__ == "__main__":
    unittest.main()

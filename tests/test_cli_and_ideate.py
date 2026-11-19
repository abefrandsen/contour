from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from contour.cli import main
from contour.mdio import write_markdown_file


class TestCliAndIdeate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        main(["init", str(self.root)])
        (self.root / "contour.yaml").write_text(yaml.safe_dump({
            "contour": 1,
            "metrics": {
                "val/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
                "sys/tput": {"direction": "max", "kind": "cost", "min_delta": 50.0},
                "attn/entropy": {"direction": "max", "kind": "diagnostic"},
            },
        }))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--root", str(self.root), "--json", *argv])
        return json.loads(buf.getvalue())

    def _seed_run(self, rid: str, **overrides) -> None:
        fm = {
            "title": f"Run {rid}",
            "status": "final",
            "created_at": "2026-11-16T10:00:00Z",
            "config": {"optim": {"lr": 0.001, "wd": 0.05}},
            "compute": {"gpu_hours": 10.0, "params": 22_000_000},
            "code": {"commit": "c1", "dirty": False},
            "launch_cmd": "python train.py",
            "metrics": {"val/acc": 80.0, "sys/tput": 1000.0, "attn/entropy": 1.5},
        }
        fm.update(overrides)
        write_markdown_file(
            self.root / "runs" / f"{rid}.md",
            fm,
            "## Key Takeaway (TL;DR)\n- Summary.\n\n## Observations & Anomalies\n- Tail loss drops slowly.\n",
        )

    def test_end_to_end_config_diff_ideate_moves_and_cli(self) -> None:
        self._seed_run("run_20261116_00000001")
        # Sibling 1: clean marginal changing only optim.lr (+0.30 < min_delta 0.50)
        self._seed_run(
            "run_20261116_00000002",
            parent="[[run_20261116_00000001]]",
            config={"optim": {"lr": 0.002, "wd": 0.05}},
            metrics={"val/acc": 80.30, "sys/tput": 1000.0, "attn/entropy": 1.6},
        )
        # Sibling 2: clean marginal changing only optim.wd (+0.35 < min_delta 0.50) -> disjoint keys, sum=+0.65!
        self._seed_run(
            "run_20261116_00000003",
            parent="[[run_20261116_00000001]]",
            config={"optim": {"lr": 0.001, "wd": 0.1}},
            metrics={"val/acc": 80.35, "sys/tput": 1000.0, "attn/entropy": 1.7},
        )
        # Multi-unit clean win changing 2 separate blocks (optim.lr + drop) for Move 4 Occam
        self._seed_run(
            "run_20261116_00000004",
            parent="[[run_20261116_00000001]]",
            config={"optim": {"lr": 0.003, "wd": 0.05}, "drop": 0.1},
            compute={"gpu_hours": 9.5, "params": 22_000_000},
            metrics={"val/acc": 80.80, "sys/tput": 1050.0, "attn/entropy": 1.8},
        )
        # Exploratory (dirty) positive run for Move 1
        self._seed_run(
            "run_20261116_00000005",
            parent="[[run_20261116_00000001]]",
            code={"commit": "c1", "dirty": True},
            config={"optim": {"lr": 0.004, "wd": 0.05}},
            metrics={"val/acc": 81.20, "sys/tput": 1000.0},
        )

        check_out = self._run_json(["check"])
        self.assertEqual(check_out["issues"], [])

        main(["--root", str(self.root), "new", "goal", "vit_opt", "--title", "Optimize ViT"])
        main(["--root", str(self.root), "new", "claim", "lr_wd", "--title", "LR and WD help"])

        link_m = self._run_json(["link", "run_20261116_00000002", "clm_lr_wd"])
        self.assertEqual(link_m["relation"], "qualifies")
        link_w = self._run_json(["link", "run_20261116_00000004", "clm_lr_wd"])
        self.assertEqual(link_w["relation"], "supports")

        ideate_out = self._run_json(["ideate"])
        self.assertEqual(len(ideate_out["move_1_investigate_flagged"]), 1)
        self.assertEqual(ideate_out["move_1_investigate_flagged"][0]["run"], "run_20261116_00000005")

        m3 = ideate_out["move_3_combine_wins"]
        marginal_pair = [
            p for p in m3
            if {p["left"]["id"], p["right"]["id"]} == {"run_20261116_00000002", "run_20261116_00000003"}
        ]
        self.assertEqual(len(marginal_pair), 1)
        self.assertTrue(marginal_pair[0]["crosses_threshold"])
        self.assertAlmostEqual(marginal_pair[0]["projected_primary_delta"], 0.65)
        self.assertEqual(marginal_pair[0]["combined_keys"], ["optim.lr", "optim.wd"])

        m4_runs = {x["run"] for x in ideate_out["move_4_strip_complexity"]}
        self.assertIn("run_20261116_00000004", m4_runs)
        self.assertEqual(ideate_out["move_6_diagnose_frontier"][0]["best_run"], "run_20261116_00000005")

        runs_clean = self._run_json(["runs", "--control", "clean", "--status", "final"])
        self.assertEqual(runs_clean["count"], 4)
        cmp_out = self._run_json(["compare", "run_20261116_00000002", "--vs", "run_20261116_00000004"])
        self.assertEqual(cmp_out["verdict"], "win")

        st = self._run_json(["status"])
        self.assertEqual(st["counts"]["runs"], 5)
        sh = self._run_json(["show", "run_20261116_00000004"])
        self.assertEqual(sh["frontmatter"]["vs_parent"]["verdict"], "win")

    def test_init_creates_dirs_gitignore_and_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--json", "init", str(repo / "contour"), "--with-skill"])
            out = json.loads(buf.getvalue())
            croot = repo / "contour"
            self.assertTrue((croot / "contour.yaml").exists())
            self.assertTrue((croot / "runs" / "configs").is_dir())
            self.assertTrue((croot / "analyses" / "scripts").is_dir())
            self.assertTrue((croot / "analyses" / "artifacts").is_dir())
            self.assertFalse((croot / "analyses" / "configs").exists())
            gi = (croot / ".gitignore").read_text(encoding="utf-8")
            self.assertIn("analyses/scripts/", gi)
            self.assertIn("analyses/artifacts/", gi)
            self.assertTrue((repo / ".agent" / "skills" / "contour-research" / "SKILL.md").exists())
            self.assertIn("root", out)

    def test_ideate_goal_scoping_and_multi_frontier(self) -> None:
        (self.root / "contour.yaml").write_text(yaml.safe_dump({
            "contour": 1,
            "metrics": {
                "val/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
                "ood/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
                "sys/tput": {"direction": "max", "kind": "cost", "min_delta": 50.0},
            },
        }, sort_keys=False))
        main(["--root", str(self.root), "new", "goal", "g1", "--title", "Goal 1"])
        main(["--root", str(self.root), "new", "goal", "g2", "--title", "Goal 2"])
        self._seed_run("run_20261116_00000001", motivated_by=["[[goal_g1]]"])
        self._seed_run(
            "run_20261116_00000002",
            parent="[[run_20261116_00000001]]",
            motivated_by=["[[goal_g1]]"],
            code={"commit": "c1", "dirty": True},
            metrics={"val/acc": 81.5, "ood/acc": 72.0, "sys/tput": 1000.0},
        )
        self._seed_run("run_20261116_00000008", motivated_by=["[[goal_g2]]"], metrics={"val/acc": 75.0})
        self._seed_run(
            "run_20261116_00000009",
            parent="[[run_20261116_00000008]]",
            motivated_by=["[[goal_g2]]"],
            code={"commit": "c1", "dirty": True},
            metrics={"val/acc": 77.0},
        )

        ideate_g1 = self._run_json(["ideate", "--goal", "goal_g1"])
        m1_runs = {x["run"] for x in ideate_g1["move_1_investigate_flagged"]}
        self.assertEqual(m1_runs, {"run_20261116_00000002"})

        frontier_out = self._run_json(["frontier"])
        self.assertEqual(frontier_out["metric"], "val/acc")
        self.assertEqual(len(frontier_out["additional_frontiers"]), 1)
        self.assertEqual(frontier_out["additional_frontiers"][0]["metric"], "ood/acc")


if __name__ == "__main__":
    unittest.main()

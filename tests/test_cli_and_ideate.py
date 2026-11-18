from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from contour.cli import main
from contour.mdio import parse_markdown_file, write_markdown_file


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
            "id": rid,
            "type": "run",
            "title": f"Run {rid}",
            "validity": "ok",
            "interventions": [],
            "intervention_note": None,
            "parent_runs": [],
            "baseline": None,
            "motivated_by": [],
            "status": "final",
            "created_at": "2026-11-16T10:00:00Z",
            "finished_at": "2026-11-16T12:00:00Z",
            "task": "cls",
            "arch_family": "vit",
            "arch_scale": "22M",
            "dataset_train": ["in1k"],
            "tier": "proxy",
            "config": {"lr": 0.001, "wd": 0.05},
            "compute": {"gpu_hours": 10.0, "params": 22_000_000},
            "code": {"commit": "c1", "dirty": False},
            "launch_cmd": "python train.py",
            "metrics": {"val/acc": 80.0, "sys/tput": 1000.0, "attn/entropy": 1.5},
            "vs_parent": None,
        }
        fm.update(overrides)
        write_markdown_file(
            self.root / "runs" / f"{rid}.md",
            fm,
            "## Key Takeaway (TL;DR)\n- Summary.\n\n## Observations & Anomalies\n- Tail loss drops slowly.\n",
        )

    def test_end_to_end_compounding_marginal_wins_and_moves(self) -> None:
        self._seed_run("run_20261116_00000001")
        # Sibling 1: clean marginal (+0.30 < min_delta 0.50)
        self._seed_run(
            "run_20261116_00000002",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["lr=0.002"],
            config={"lr": 0.002, "wd": 0.05},
            metrics={"val/acc": 80.30, "sys/tput": 1000.0, "attn/entropy": 1.6},
        )
        # Sibling 2: clean marginal (+0.35 < min_delta 0.50) -> sum = +0.65 >= 0.50!
        self._seed_run(
            "run_20261116_00000003",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["wd=0.1"],
            config={"lr": 0.001, "wd": 0.1},
            metrics={"val/acc": 80.35, "sys/tput": 1000.0, "attn/entropy": 1.7},
        )
        # Multi-tag win for Move 4 Occam
        self._seed_run(
            "run_20261116_00000004",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["lr=0.003", "wd=0.2"],
            config={"lr": 0.003, "wd": 0.2},
            compute={"gpu_hours": 9.5, "params": 22_000_000},
            metrics={"val/acc": 80.80, "sys/tput": 1050.0, "attn/entropy": 1.8},
        )
        # Confounded positive run for Move 1
        self._seed_run(
            "run_20261116_00000005",
            parent_runs=["[[run_20261116_00000001]]"],
            dataset_train=["in1k_plus"],
            interventions=["lr=0.004"],
            config={"lr": 0.004, "wd": 0.05},
            metrics={"val/acc": 81.20, "sys/tput": 1000.0},
        )

        check_out = self._run_json(["check", "--fix"])
        self.assertEqual(len(check_out["fixed"]), 4)
        self.assertEqual(check_out["issues"], [])

        # Goal + Claim creation and linking
        main(["--root", str(self.root), "new", "goal", "vit_opt", "--title", "Optimize ViT"])
        main(["--root", str(self.root), "new", "claim", "lr_wd", "--title", "LR and WD help"])

        # Advisory link for marginal run -> qualifies; for win -> supports
        link_m = self._run_json(["link", "run_20261116_00000002", "clm_lr_wd"])
        self.assertEqual(link_m["relation"], "qualifies")
        link_w = self._run_json(["link", "run_20261116_00000004", "clm_lr_wd"])
        self.assertEqual(link_w["relation"], "supports")

        # Ideate detects Move 1, Move 3 compounding marginal pair crossing tau, Move 4, Move 6
        ideate_out = self._run_json(["ideate"])
        self.assertEqual(len(ideate_out["move_1_isolate_confounds"]), 1)
        self.assertEqual(ideate_out["move_1_isolate_confounds"][0]["run"], "run_20261116_00000005")

        m3 = ideate_out["move_3_combine_wins"]
        marginal_pair = [
            p for p in m3
            if {p["left"]["id"], p["right"]["id"]} == {"run_20261116_00000002", "run_20261116_00000003"}
        ]
        self.assertEqual(len(marginal_pair), 1)
        self.assertTrue(marginal_pair[0]["crosses_threshold"])
        self.assertAlmostEqual(marginal_pair[0]["projected_primary_delta"], 0.65)

        self.assertEqual(len(ideate_out["move_4_strip_complexity"]), 1)
        self.assertEqual(ideate_out["move_6_diagnose_frontier"][0]["best_run"], "run_20261116_00000005")

        # Runs query and compare
        runs_clean = self._run_json(["runs", "--control", "clean"])
        self.assertEqual(runs_clean["count"], 3)
        cmp_out = self._run_json(["compare", "run_20261116_00000002", "--vs", "run_20261116_00000004"])
        self.assertEqual(cmp_out["verdict"], "win")

        # Status & show
        st = self._run_json(["status"])
        self.assertEqual(st["counts"]["runs"], 5)
        sh = self._run_json(["show", "run_20261116_00000004"])
        self.assertEqual(sh["frontmatter"]["vs_parent"]["verdict"], "win")

    def test_init_with_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--json", "init", str(repo / "contour"), "--with-skill"])
            out = json.loads(buf.getvalue())
            self.assertTrue((repo / "contour" / "contour.yaml").exists())
            self.assertTrue((repo / ".agent" / "skills" / "contour-research" / "SKILL.md").exists())
            self.assertIn("root", out)

    def test_ideate_move1_move3_move4_edge_cases_and_multi_frontier(self) -> None:
        (self.root / "contour.yaml").write_text(yaml.safe_dump({
            "contour": 1,
            "metrics": {
                "val/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
                "ood/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.50},
                "sys/tput": {"direction": "max", "kind": "cost", "min_delta": 50.0},
            },
        }, sort_keys=False))
        self._seed_run("run_20261116_00000001")
        # Unrelated facet-matched run with different intervention (wd=0.2) must NOT be suggested as Move 1 bridge
        self._seed_run(
            "run_20261116_00000002",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["wd=0.2"],
            config={"lr": 0.001, "wd": 0.2},
            metrics={"val/acc": 80.1, "sys/tput": 1000.0},
        )
        # Confounded positive run (dataset_train shift + lr=0.002, wd=0.1)
        self._seed_run(
            "run_20261116_00000003",
            parent_runs=["[[run_20261116_00000001]]"],
            dataset_train=["in1k_v2"],
            interventions=["lr=0.002", "wd=0.1"],
            config={"lr": 0.002, "wd": 0.1},
            metrics={"val/acc": 81.0, "sys/tput": 1000.0},
        )
        # Siblings on 00000001 that conflict on `lr` (lr=0.002 vs lr=0.003) must NOT combine in Move 3
        self._seed_run(
            "run_20261116_00000004",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["lr=0.002"],
            config={"lr": 0.002, "wd": 0.05},
            metrics={"val/acc": 80.3, "sys/tput": 1000.0},
        )
        self._seed_run(
            "run_20261116_00000005",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["lr=0.003"],
            config={"lr": 0.003, "wd": 0.05},
            metrics={"val/acc": 80.4, "sys/tput": 1000.0},
        )
        # Multi-tag win with a downstream synthesis child must STILL appear in Move 4
        self._seed_run(
            "run_20261116_00000006",
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["lr=0.004", "wd=0.3"],
            config={"lr": 0.004, "wd": 0.3},
            metrics={"val/acc": 80.8, "sys/tput": 1000.0},
        )
        self._seed_run(
            "run_20261116_00000007",
            parent_runs=["[[run_20261116_00000006]]"],
            interventions=["drop=0.1"],
            config={"lr": 0.004, "wd": 0.3, "drop": 0.1},
            metrics={"val/acc": 81.2, "ood/acc": 72.0, "sys/tput": 1000.0},
        )
        # Unrelated chain on another root with combined tags {lr=0.002, wd=0.2} must not suppress 00000002+00000004
        self._seed_run(
            "run_20261116_00000008",
            dataset_train=["other"],
            config={"lr": 0.001, "wd": 0.05},
            metrics={"val/acc": 75.0, "sys/tput": 1000.0},
        )
        self._seed_run(
            "run_20261116_00000009",
            parent_runs=["[[run_20261116_00000008]]"],
            dataset_train=["other"],
            interventions=["lr=0.002", "wd=0.2"],
            config={"lr": 0.002, "wd": 0.2},
            metrics={"val/acc": 76.0, "sys/tput": 1000.0},
        )

        ideate_out = self._run_json(["ideate"])
        m1 = [x for x in ideate_out["move_1_isolate_confounds"] if x["run"] == "run_20261116_00000003"]
        self.assertEqual(len(m1), 1)
        self.assertEqual(m1[0]["bridge_candidates"], [])

        m3_pairs = [tuple(sorted([p["left"]["id"], p["right"]["id"]])) for p in ideate_out["move_3_combine_wins"]]
        self.assertNotIn(("run_20261116_00000004", "run_20261116_00000005"), m3_pairs)
        self.assertIn(("run_20261116_00000002", "run_20261116_00000004"), m3_pairs)

        m4_runs = {x["run"] for x in ideate_out["move_4_strip_complexity"]}
        self.assertIn("run_20261116_00000006", m4_runs)

        frontier_out = self._run_json(["frontier"])
        self.assertEqual(frontier_out["metric"], "val/acc")
        self.assertEqual(len(frontier_out["additional_frontiers"]), 1)
        self.assertEqual(frontier_out["additional_frontiers"][0]["metric"], "ood/acc")


if __name__ == "__main__":
    unittest.main()

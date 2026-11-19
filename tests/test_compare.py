import tempfile
from pathlib import Path
import unittest

import yaml

from contour.cli import main
from contour.compare import compare_runs, compute_vs_parent
from contour.config import load_config
from contour.graph import ContourGraph
from contour.mdio import write_markdown_file


class TestCompare(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        main(["init", str(self.root)])
        (self.root / "contour.yaml").write_text(yaml.safe_dump({
            "contour": 1,
            "ignored_config_keys": ["seed", "num_workers"],
            "metrics": {
                "val/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.5},
                "ood/acc": {"direction": "max", "kind": "quality", "min_delta": 0.5},
                "sys/tput": {"direction": "max", "kind": "cost", "min_rel_delta": 0.05},
                "attn/entropy": {"direction": "max", "kind": "diagnostic"},
            },
        }))
        self.cfg = load_config(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _fm(self, **overrides) -> dict:
        base = {
            "title": "Run",
            "status": "final",
            "config": {"optim": {"lr": 0.001, "wd": 0.05}, "seed": 0, "num_workers": 4},
            "compute": {"gpu_hours": 10.0, "params": 20_000_000},
            "code": {"commit": "c1", "dirty": False},
            "metrics": {"val/acc": 80.0, "ood/acc": 70.0, "sys/tput": 1000.0, "attn/entropy": 1.2},
        }
        base.update(overrides)
        return base

    def test_nested_config_diff_subtree_blocks_and_ignored_keys(self) -> None:
        base = self._fm(
            config={
                "optim": {"lr": 0.001, "wd": 0.05, "seed": 0},
                "encoder": {"type": "resnet50", "depth": 50, "width": 64},
                "num_workers": 4,
            },
        )
        # Changing only ignored keys (including nested seed) -> empty config_diff + replicate flag
        rep = self._fm(
            config={
                "optim": {"lr": 0.001, "wd": 0.05, "seed": 7},
                "encoder": {"type": "resnet50", "depth": 50, "width": 64},
                "num_workers": 8,
            },
        )
        vs_rep = compute_vs_parent(rep, base, self.cfg, run_id="run_r", base_id="run_b")
        self.assertEqual(vs_rep["config_diff"], {})
        self.assertEqual(vs_rep["control_level"], "clean")
        self.assertIn("replicate", vs_rep["flags"])

        # Wholesale encoder swap + LR change stays clean and groups encoder.* block
        swap = self._fm(
            config={
                "optim": {"lr": 0.0005, "wd": 0.05, "seed": 0},
                "encoder": {"type": "siglip_b", "patch_size": 16, "num_heads": 12},
            },
            metrics={"val/acc": 82.0, "ood/acc": 73.0, "sys/tput": 1000.0},
        )
        vs_swap = compute_vs_parent(swap, base, self.cfg, run_id="run_swap", base_id="run_b")
        self.assertEqual(vs_swap["control_level"], "clean")
        self.assertEqual(vs_swap["verdict"], "win")
        self.assertIn("optim.lr", vs_swap["config_diff"])
        self.assertIn("encoder", vs_swap["config_blocks"])
        self.assertEqual(vs_swap["config_blocks"]["encoder"]["baseline"], "resnet50")
        self.assertEqual(vs_swap["config_blocks"]["encoder"]["value"], "siglip_b")
        self.assertIn("multi_block_config_diff", vs_swap["flags"])

    def test_control_level_hard_facts_and_advisory_flags(self) -> None:
        base = self._fm()
        dirty = self._fm(code={"commit": "c1", "dirty": True})
        vs_d = compute_vs_parent(dirty, base, self.cfg, run_id="run_d", base_id="run_b")
        self.assertEqual(vs_d["control_level"], "exploratory")
        self.assertIn("code_dirty", vs_d["flags"])

        bug = self._fm(bugged=True)
        self.assertEqual(compute_vs_parent(bug, base, self.cfg, run_id="run_bug", base_id="run_b")["control_level"], "invalid")

        diff_c = self._fm(code={"commit": "c2", "dirty": False})
        vs_c2 = compute_vs_parent(diff_c, base, self.cfg, run_id="run_c2", base_id="run_b")
        self.assertEqual(vs_c2["control_level"], "clean")
        self.assertIn("code_changed", vs_c2["flags"])

    def test_verdict_table_min_rel_delta_and_compute_ratios(self) -> None:
        base = self._fm()
        # clear primary + matched cost -> win
        w = self._fm(config={"optim": {"lr": 0.002}}, metrics={"val/acc": 80.8, "ood/acc": 70.1, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(w, base, self.cfg)["verdict"], "win")

        # clear primary + non-primary regression -> tradeoff
        tr = self._fm(config={"optim": {"lr": 0.002}}, metrics={"val/acc": 80.8, "ood/acc": 69.0, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(tr, base, self.cfg)["verdict"], "tradeoff")

        # neutral primary + >=5% throughput gain (min_rel_delta: 0.05) -> efficiency_win
        ew = self._fm(
            compute={"gpu_hours": 7.0, "params": 20_000_000},
            metrics={"val/acc": 79.9, "ood/acc": 70.0, "sys/tput": 1060.0},
        )
        vs_ew = compute_vs_parent(ew, base, self.cfg)
        self.assertEqual(vs_ew["deltas"]["sys/tput"]["signal"], "clear")
        self.assertAlmostEqual(vs_ew["deltas"]["sys/tput"]["rel_delta"], 0.06)
        self.assertEqual(vs_ew["verdict"], "efficiency_win")
        self.assertAlmostEqual(vs_ew["compute_diff"]["gpu_hours"]["ratio"], 0.7)

        # >=5% throughput regression -> tradeoff
        cost_reg = self._fm(metrics={"val/acc": 81.0, "ood/acc": 70.5, "sys/tput": 940.0})
        self.assertEqual(compute_vs_parent(cost_reg, base, self.cfg)["verdict"], "tradeoff")

    def test_compare_runs_composes_attached_evals_and_flags_script_drift(self) -> None:
        r1 = self._fm(title="Base", metrics={"val/acc": 80.0})
        r2 = self._fm(title="Treat", metrics={"val/acc": 80.8})
        write_markdown_file(self.root / "runs" / "run_20261116_00000001.md", r1, "## Key Takeaway (TL;DR)\n- ok\n")
        write_markdown_file(self.root / "runs" / "run_20261116_00000002.md", r2, "## Key Takeaway (TL;DR)\n- ok\n")

        e1 = {
            "kind": "eval",
            "title": "OOD eval base",
            "status": "final",
            "subjects": ["[[run_20261116_00000001]]"],
            "script": "analyses/scripts/eval_ood.py",
            "script_sha256": "aaaa1111bbbb2222",
            "launch_cmd": "python analyses/scripts/eval_ood.py --ckpt r1",
            "code": {"commit": "e1", "dirty": False},
            "metrics": {"ood/acc": 70.0},
        }
        e2 = {
            "kind": "eval",
            "title": "OOD eval treat",
            "status": "final",
            "subjects": ["[[run_20261116_00000002]]"],
            "script": "analyses/scripts/eval_ood.py",
            "script_sha256": "cccc3333dddd4444",  # script changed!
            "launch_cmd": "python analyses/scripts/eval_ood.py --ckpt r2",
            "code": {"commit": "e1", "dirty": False},
            "metrics": {"ood/acc": 72.0},
        }
        write_markdown_file(self.root / "analyses" / "ana_20261117_00000001.md", e1, "## Key Takeaway (TL;DR)\n- ok\n")
        write_markdown_file(self.root / "analyses" / "ana_20261117_00000002.md", e2, "## Key Takeaway (TL;DR)\n- ok\n")

        g = ContourGraph(self.root)
        cmp = compare_runs(g, "run_20261116_00000001", "run_20261116_00000002")
        self.assertIn("ood/acc", cmp["deltas"])
        self.assertAlmostEqual(cmp["deltas"]["ood/acc"]["delta"], 2.0)
        self.assertIn("eval_drift:ood/acc", cmp["flags"])
        self.assertIn("eval_script_changed", cmp["flags"])
        self.assertEqual(cmp["eval_provenance"]["ood/acc"]["baseline_eval"], "ana_20261117_00000001")


if __name__ == "__main__":
    unittest.main()

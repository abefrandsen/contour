import tempfile
from pathlib import Path
import unittest

import yaml

from contour.check import fix_stale_vs_parent
from contour.cli import main
from contour.compare import compute_vs_parent
from contour.config import load_config
from contour.graph import ContourGraph
from contour.mdio import parse_markdown_file, write_markdown_file


class TestCompare(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        main(["init", str(self.root)])
        (self.root / "contour.yaml").write_text(yaml.safe_dump({
            "contour": 1,
            "iso_tolerance": {"params": 0.02, "gpu_hours": 0.10},
            "ignored_config_keys": ["seed", "num_workers"],
            "metrics": {
                "val/acc": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.5},
                "ood/acc": {"direction": "max", "kind": "quality", "min_delta": 0.5},
                "sys/tput": {"direction": "max", "kind": "cost", "min_delta": 50.0},
                "attn/entropy": {"direction": "max", "kind": "diagnostic"},
            },
        }))
        self.cfg = load_config(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _fm(self, rid: str, **overrides) -> dict:
        base = {
            "id": rid,
            "type": "run",
            "status": "final",
            "validity": "ok",
            "task": "cls",
            "arch_family": "vit",
            "arch_scale": "22M",
            "dataset_train": ["in1k"],
            "tier": "proxy",
            "interventions": [],
            "config": {"lr": 0.001, "seed": 0, "num_workers": 4},
            "compute": {"gpu_hours": 10.0, "params": 20_000_000},
            "code": {"commit": "c1", "dirty": False},
            "metrics": {"val/acc": 80.0, "ood/acc": 70.0, "sys/tput": 1000.0, "attn/entropy": 1.2},
        }
        base.update(overrides)
        return base

    def test_diff_accountability_and_ignored_keys(self) -> None:
        base = self._fm("run_b")
        # Changing seed & num_workers with no interventions -> replicate (exploratory)
        rep = self._fm("run_r", config={"lr": 0.001, "seed": 1, "num_workers": 8})
        vs_rep = compute_vs_parent(rep, base, self.cfg)
        self.assertEqual(vs_rep["config_diff"], {})
        self.assertEqual(vs_rep["control_level"], "exploratory")

        # Undeclared lr change -> unaccounted -> confounded
        conf = self._fm("run_c", config={"lr": 0.002, "seed": 0})
        vs_conf = compute_vs_parent(conf, base, self.cfg)
        self.assertEqual(vs_conf["control_level"], "confounded")
        self.assertEqual(vs_conf["unaccounted"], ["lr"])

        # Declared key=value mismatch -> unrealized -> confounded
        unr = self._fm("run_u", interventions=["lr=0.003"], config={"lr": 0.002, "seed": 0})
        vs_unr = compute_vs_parent(unr, base, self.cfg)
        self.assertEqual(vs_unr["control_level"], "confounded")
        self.assertEqual(vs_unr["unrealized"], ["lr=0.003"])

        # Free-text code intervention on same clean commit -> unrealized -> confounded
        same_commit = self._fm("run_f", interventions=["rope"])
        vs_same = compute_vs_parent(same_commit, base, self.cfg)
        self.assertEqual(vs_same["control_level"], "confounded")
        self.assertEqual(vs_same["unrealized"], ["rope"])

        # Free-text code intervention on new commit -> clean
        new_commit = self._fm("run_n", interventions=["rope"], code={"commit": "c2", "dirty": False})
        vs_new = compute_vs_parent(new_commit, base, self.cfg)
        self.assertEqual(vs_new["control_level"], "clean")

    def test_q_x_c_verdict_table_and_non_primary_downgrade(self) -> None:
        base = self._fm("run_b")
        # clear primary + matched cost -> win
        w = self._fm("run_w", interventions=["lr=0.002"], config={"lr": 0.002},
                     metrics={"val/acc": 80.8, "ood/acc": 70.1, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(w, base, self.cfg)["verdict"], "win")

        # clear primary + non-primary regression -> tradeoff
        tr = self._fm("run_tr", interventions=["lr=0.002"], config={"lr": 0.002},
                      metrics={"val/acc": 80.8, "ood/acc": 69.0, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(tr, base, self.cfg)["verdict"], "tradeoff")

        # neutral primary + reduced gpu_hours -> efficiency_win
        ew = self._fm("run_ew", interventions=["lr=0.002"], config={"lr": 0.002},
                      compute={"gpu_hours": 7.0, "params": 20_000_000},
                      metrics={"val/acc": 79.9, "ood/acc": 70.0, "sys/tput": 1000.0})
        vs_ew = compute_vs_parent(ew, base, self.cfg)
        self.assertEqual(vs_ew["iso_compute"], "reduced")
        self.assertEqual(vs_ew["verdict"], "efficiency_win")

        # marginal primary (0 < 0.2 < 0.5) + increased compute -> tradeoff
        mg_bad = self._fm("run_mg", interventions=["lr=0.002"], config={"lr": 0.002},
                          compute={"gpu_hours": 15.0, "params": 20_000_000},
                          metrics={"val/acc": 80.2, "ood/acc": 70.0, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(mg_bad, base, self.cfg)["verdict"], "tradeoff")

        # partial compute fields: only gpu_hours present, params missing -> still works
        partial = self._fm("run_p", interventions=["lr=0.002"], config={"lr": 0.002},
                           compute={"gpu_hours": 10.0},
                           metrics={"val/acc": 80.8, "ood/acc": 70.0, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(partial, base, self.cfg)["iso_compute"], "matched")

        # zero gpu_hours on both -> matched (not unknown); zero -> positive -> increased
        z_base = self._fm("run_zb", compute={"gpu_hours": 0.0})
        z_same = self._fm("run_zs", interventions=["lr=0.002"], config={"lr": 0.002},
                          compute={"gpu_hours": 0.0},
                          metrics={"val/acc": 80.8, "ood/acc": 70.0, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(z_same, z_base, self.cfg)["iso_compute"], "matched")
        z_more = self._fm("run_zm", interventions=["lr=0.002"], config={"lr": 0.002},
                          compute={"gpu_hours": 1.0},
                          metrics={"val/acc": 80.8, "ood/acc": 70.0, "sys/tput": 1000.0})
        self.assertEqual(compute_vs_parent(z_more, z_base, self.cfg)["iso_compute"], "increased")

    def test_salvage_via_baseline_override_and_check_fix(self) -> None:
        base_in1k = self._fm("run_20261116_00000001", dataset_train=["in1k"])
        base_web = self._fm("run_20261116_00000002", dataset_train=["web"])
        treat = self._fm(
            "run_20261116_00000003",
            dataset_train=["web"],
            parent_runs=["[[run_20261116_00000001]]"],
            interventions=["lr=0.002"],
            config={"lr": 0.002},
            metrics={"val/acc": 81.0, "ood/acc": 70.5, "sys/tput": 1000.0},
        )
        for d in (base_in1k, base_web, treat):
            write_markdown_file(self.root / "runs" / f"{d['id']}.md", d, "## Key Takeaway (TL;DR)\n- ok\n")

        g = ContourGraph(self.root)
        fix_stale_vs_parent(g)
        fm, _ = parse_markdown_file(self.root / "runs" / "run_20261116_00000003.md")
        self.assertEqual(fm["vs_parent"]["control_level"], "confounded")

        # Salvage by pointing baseline: to the web baseline while preserving parent_runs lineage
        fm["baseline"] = "[[run_20261116_00000002]]"
        write_markdown_file(self.root / "runs" / "run_20261116_00000003.md", fm, "## Key Takeaway (TL;DR)\n- ok\n")
        g = ContourGraph(self.root)
        fix_stale_vs_parent(g)
        fm_fixed, _ = parse_markdown_file(self.root / "runs" / "run_20261116_00000003.md")
        self.assertEqual(fm_fixed["vs_parent"]["control_level"], "clean")
        self.assertEqual(fm_fixed["vs_parent"]["verdict"], "win")
        self.assertEqual(fm_fixed["parent_runs"], ["[[run_20261116_00000001]]"])


if __name__ == "__main__":
    unittest.main()

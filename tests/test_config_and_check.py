import datetime as dt
import tempfile
from pathlib import Path
import unittest

import yaml

from contour.check import (
    compute_frontier,
    compute_verified_hash,
    effective_verification,
    find_check_issues,
    find_review_items,
    fix_stale_vs_parent,
    issues_blocking_verify,
)
from contour.cli import main
from contour.config import ConfigError, find_root, load_config, validate_config
from contour.graph import ContourGraph
from contour.mdio import parse_markdown_file, write_markdown_file


def _write_run(
    root: Path,
    rid: str,
    *,
    status: str = "final",
    validity: str = "ok",
    tier: str = "proxy",
    parent_runs: list[str] | None = None,
    baseline: str | None = None,
    motivated_by: list[str] | None = None,
    interventions: list[str] | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
    compute: dict | None = None,
    code: dict | None = None,
    created_at: str = "2026-11-16T10:00:00Z",
    body: str = "## Key Takeaway (TL;DR)\n- Concrete summary.\n",
) -> None:
    fm = {
        "id": rid,
        "type": "run",
        "title": f"Run {rid}",
        "validity": validity,
        "interventions": interventions or [],
        "intervention_note": None,
        "parent_runs": [f"[[{p}]]" for p in (parent_runs or [])],
        "baseline": f"[[{baseline}]]" if baseline else None,
        "motivated_by": [f"[[{m}]]" for m in (motivated_by or [])],
        "status": status,
        "created_at": created_at,
        "finished_at": "2026-11-16T12:00:00Z" if status != "running" else None,
        "task": "lm",
        "arch_family": "gpt",
        "arch_scale": "125M",
        "dataset_train": ["pile"],
        "tier": tier,
        "config": config or {"lr": 1e-3},
        "compute": compute or {"gpu_hours": 10.0, "params": 125_000_000},
        "code": code or {"commit": "abc123", "dirty": False},
        "launch_cmd": "python train.py",
        "metrics": metrics if metrics is not None else {"val/loss": 3.0},
        "vs_parent": None,
    }
    write_markdown_file(root / "runs" / f"{rid}.md", fm, body)


class TestConfigAndCheck(unittest.TestCase):
    def test_find_root_discovers_contour_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / "contour").mkdir()
            (repo / "contour" / "contour.yaml").write_text("contour: 1\nmetrics: {}\n")
            nested = repo / "src" / "models"
            nested.mkdir(parents=True)
            self.assertEqual(find_root(start=nested), repo / "contour")

    def test_config_validation_rules(self) -> None:
        errs = validate_config({
            "contour": 1,
            "metrics": {
                "badkey": {"direction": "up", "kind": "quality"},
                "val/loss": {"direction": "min", "kind": "quality"},  # missing min_delta
                "attn/entropy": {"direction": "max", "kind": "diagnostic"},  # ok without min_delta
            },
        })
        self.assertTrue(any("'<scope>/<name>'" in e for e in errs))
        self.assertTrue(any("direction" in e for e in errs))
        self.assertTrue(any("min_delta is required" in e for e in errs))
        self.assertFalse(any("attn/entropy" in e for e in errs))

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "contour.yaml").write_text("contour: 99\n")
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_check_rules_c001_through_c009(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            (root / "contour.yaml").write_text(yaml.safe_dump({
                "contour": 1,
                "metrics": {
                    "val/loss": {"direction": "min", "kind": "quality", "primary": True, "min_delta": 0.05},
                },
            }))

            _write_run(root, "run_20261116_00000001", metrics={"val/loss": 3.0})
            # C004 placeholder + C006 stale vs_parent + C009 undeclared metric
            _write_run(
                root,
                "run_20261116_00000002",
                parent_runs=["run_20261116_00000001"],
                interventions=["lr=0.002"],
                config={"lr": 0.002},
                metrics={"val/loss": 2.9, "BadMetric": 1.0},
                motivated_by=["clm_test"],
                body="## Key Takeaway (TL;DR)\n- TODO: fill me in.\n",
            )
            # C002 non-usable run (bugged)
            _write_run(
                root,
                "run_20261116_00000003",
                validity="bugged",
                metrics={"val/loss": 2.5},
            )
            # C005 stuck running > 72h
            _write_run(
                root,
                "run_20261116_00000004",
                status="running",
                created_at="2026-11-10T00:00:00Z",
                metrics={},
            )

            # Claim with: C001 missing goal + duplicate edge, C002 bugged run, C003 uncited run,
            # C004 TODO, C007 supported with non-clean/missing run
            write_markdown_file(
                root / "claims" / "clm_test.md",
                {
                    "id": "clm_test",
                    "type": "claim",
                    "title": "Test claim",
                    "status": "supported",
                    "superseded_by": None,
                    "verification": "draft",
                    "verified_hash": None,
                    "audit_note": None,
                    "goals": ["[[goal_missing]]"],
                    "supports": ["[[run_20261116_00000003]]"],
                    "contradicts": ["[[run_20261116_00000003]]"],
                    "qualifies": [],
                    "scope": {"holds_in": [], "breaks_in": [], "untested": []},
                },
                "## Hypothesis & Mechanism\n- TODO: explain\n\n## Evidence Synthesis\n- missing citation\n",
            )

            g = ContourGraph(root)
            now = dt.datetime(2026, 11, 18, 0, 0, tzinfo=dt.timezone.utc)
            codes = {i.code for i in find_check_issues(g, now=now)}
            for expected in ("C001", "C002", "C003", "C004", "C005", "C006", "C007", "C009"):
                self.assertIn(expected, codes)

            # check --fix repairs C006 without touching author body
            fixed = fix_stale_vs_parent(g)
            self.assertEqual(fixed, ["run_20261116_00000002"])
            fm2, body2 = parse_markdown_file(root / "runs" / "run_20261116_00000002.md")
            self.assertEqual(fm2["vs_parent"]["verdict"], "win")
            self.assertIn("- TODO: fill me in.", body2)

    def test_c008_verification_staleness_and_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            (root / "contour.yaml").write_text(yaml.safe_dump({
                "contour": 1,
                "metrics": {
                    "val/loss": {"direction": "min", "kind": "quality", "primary": True, "min_delta": 0.05},
                },
            }))
            _write_run(
                root, "run_20261116_00000001",
                code={"commit": "c1", "dirty": False},
                metrics={"val/loss": 3.0},
            )
            _write_run(
                root, "run_20261116_00000002",
                parent_runs=["run_20261116_00000001"],
                interventions=["lr=0.002"],
                config={"lr": 0.002},
                code={"commit": "c1", "dirty": False},
                metrics={"val/loss": 2.8},
            )
            _write_run(
                root, "run_20261116_00000003",
                parent_runs=["run_20261116_00000001"],
                interventions=["dirty_exp"],
                config={"lr": 0.001},
                code={"commit": "c1", "dirty": True},
                compute={"gpu_hours": 5.0, "params": 125_000_000},
                metrics={"val/loss": 2.7},
            )
            write_markdown_file(
                root / "claims" / "clm_v.md",
                {
                    "id": "clm_v",
                    "type": "claim",
                    "title": "Verified claim",
                    "status": "supported",
                    "superseded_by": None,
                    "verification": "draft",
                    "verified_hash": None,
                    "audit_note": None,
                    "goals": [],
                    "supports": ["[[run_20261116_00000002]]"],
                    "contradicts": [],
                    "qualifies": [],
                    "scope": {"holds_in": ["lm"], "breaks_in": [], "untested": []},
                },
                "## Hypothesis & Mechanism\nLower loss.\n\n## Evidence Synthesis\n- [[run_20261116_00000002]] wins.\n",
            )

            g = ContourGraph(root)
            fix_stale_vs_parent(g)
            g = ContourGraph(root)
            self.assertEqual(issues_blocking_verify(g, "clm_v"), [])

            main(["--root", str(root), "verify", "clm_v", "--agent", "--note", "Looks solid"])
            g = ContourGraph(root)
            self.assertEqual(effective_verification(g, g.claims["clm_v"]), "agent_audited")

            rev = find_review_items(g)
            self.assertEqual(len(rev["awaiting_signoff"]), 1)
            self.assertEqual(rev["dirty_frontier"][0]["id"], "run_20261116_00000003")

            # Mutating the supporting run invalidates verified_hash -> C008 & stale_verified
            fm2, body2 = parse_markdown_file(root / "runs" / "run_20261116_00000002.md")
            fm2["metrics"]["val/loss"] = 2.85
            write_markdown_file(root / "runs" / "run_20261116_00000002.md", fm2, body2)

            g = ContourGraph(root)
            self.assertEqual(effective_verification(g, g.claims["clm_v"]), "draft")
            codes = {i.code for i in find_check_issues(g)}
            self.assertIn("C008", codes)
            rev2 = find_review_items(g)
            self.assertEqual(len(rev2["stale_verified"]), 1)

    def test_frontier_tie_break_c009_missing_min_delta_and_multi_benchmark_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            (root / "contour.yaml").write_text(yaml.safe_dump({
                "contour": 1,
                "metrics": {
                    "in1k/top1": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.2},
                    "cifar/top1": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.3},
                    "val/loss": {"direction": "min", "kind": "quality"},  # missing min_delta -> C009
                },
            }))
            # Tie on in1k/top1 (80.0): run_a is slower (20h) and alphabetically earlier than run_b (5h)
            _write_run(
                root, "run_20261116_0000000a",
                compute={"gpu_hours": 20.0, "params": 1000},
                metrics={"in1k/top1": 80.0},
            )
            _write_run(
                root, "run_20261116_0000000b",
                compute={"gpu_hours": 5.0, "params": 1000},
                metrics={"in1k/top1": 80.0},
            )
            # Same task ("lm"), second primary benchmark (cifar/top1), dirty frontier run
            _write_run(
                root, "run_20261116_0000000c",
                compute={"gpu_hours": 2.0, "params": 1000},
                code={"commit": "c1", "dirty": True},
                metrics={"cifar/top1": 74.0},
            )

            g = ContourGraph(root)
            codes = {i.code for i in find_check_issues(g)}
            self.assertIn("C009", codes)

            _, _, _, pareto_ids = compute_frontier(g, task="lm", metric_filter="in1k/top1")
            self.assertEqual(pareto_ids, {"run_20261116_0000000b"})

            rev = find_review_items(g)
            dirty_ids = {item["id"] for item in rev["dirty_frontier"]}
            self.assertIn("run_20261116_0000000c", dirty_ids)


if __name__ == "__main__":
    unittest.main()

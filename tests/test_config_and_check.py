import datetime as dt
import tempfile
from pathlib import Path
import unittest

import yaml

from contour.check import (
    ERROR,
    WARN,
    compute_frontier,
    effective_verification,
    find_check_issues,
    find_review_items,
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
    bugged: bool = False,
    parent: str | None = None,
    motivated_by: list[str] | None = None,
    config: dict | None = None,
    config_path: str | None = None,
    metrics: dict | None = None,
    compute: dict | None = None,
    code: dict | None = None,
    artifacts: list | None = None,
    created_at: str = "2026-11-16T10:00:00Z",
    body: str = "## Key Takeaway (TL;DR)\n- Concrete summary.\n",
) -> None:
    fm: dict = {
        "title": f"Run {rid}",
        "status": status,
        "created_at": created_at,
        "compute": compute if compute is not None else {"gpu_hours": 10.0, "params": 125_000_000},
        "code": code or {"commit": "abc123", "dirty": False},
        "launch_cmd": "python train.py",
        "metrics": metrics if metrics is not None else {"val/loss": 3.0},
    }
    if bugged:
        fm["bugged"] = True
    if parent:
        fm["parent"] = f"[[{parent}]]"
    if motivated_by:
        fm["motivated_by"] = [f"[[{m}]]" for m in motivated_by]
    if config_path:
        fm["config_path"] = config_path
    else:
        fm["config"] = config if config is not None else {"lr": 1e-3}
    if artifacts:
        fm["artifacts"] = artifacts
    write_markdown_file(root / "runs" / f"{rid}.md", fm, body)


def _write_analysis(
    root: Path,
    aid: str,
    *,
    kind: str = "analysis",
    status: str = "final",
    bugged: bool = False,
    subjects: list[str] | None = None,
    motivated_by: list[str] | None = None,
    code: dict | None = None,
    metrics: dict | None = None,
    artifacts: list | None = None,
    body: str = "## Key Takeaway (TL;DR)\n- Clear diagnostic finding.\n",
) -> None:
    fm: dict = {
        "kind": kind,
        "title": f"Analysis {aid}",
        "status": status,
        "subjects": [f"[[{s}]]" for s in (subjects or [])],
        "code": code or {"commit": "abc123", "dirty": False},
        "metrics": metrics or {},
    }
    if bugged:
        fm["bugged"] = True
    if motivated_by:
        fm["motivated_by"] = [f"[[{m}]]" for m in motivated_by]
    if artifacts:
        fm["artifacts"] = artifacts
    write_markdown_file(root / "analyses" / f"{aid}.md", fm, body)


class TestConfigAndCheck(unittest.TestCase):
    def test_find_root_discovers_contour_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / "contour").mkdir()
            (repo / "contour" / "contour.yaml").write_text("contour: 1\nmetrics: {}\n")
            nested = repo / "src" / "models"
            nested.mkdir(parents=True)
            self.assertEqual(find_root(start=nested), repo / "contour")

    def test_config_validation_rules_with_min_rel_delta(self) -> None:
        errs = validate_config({
            "contour": 1,
            "metrics": {
                "badkey": {"direction": "up", "kind": "quality"},
                "val/loss": {"direction": "min", "kind": "quality"},  # missing min_delta & min_rel_delta
                "sys/tput": {"direction": "max", "kind": "cost", "min_rel_delta": 0.05},  # ok!
                "attn/entropy": {"direction": "max", "kind": "diagnostic"},  # ok without threshold
            },
        })
        self.assertTrue(any("'<scope>/<name>'" in e for e in errs))
        self.assertTrue(any("direction" in e for e in errs))
        self.assertTrue(any("min_delta" in e for e in errs))
        self.assertFalse(any("sys/tput" in e for e in errs))
        self.assertFalse(any("attn/entropy" in e for e in errs))

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "contour.yaml").write_text("contour: 99\n")
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_check_rules_c001_through_c010_and_sidecar_validation(self) -> None:
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
            _write_run(
                root,
                "run_20261116_00000002",
                parent="run_20261116_00000001",
                config_path="runs/configs/missing.yaml",
                metrics={"val/loss": 2.9, "BadMetric": 1.0},
                artifacts=[{"path": "analyses/artifacts/missing.png", "kind": "plot"}, {"path": "s3://ok/ckpt.pt"}],
                motivated_by=["clm_test"],
                body="## Key Takeaway (TL;DR)\n- TODO: fill me in.\n",
            )
            # Quarantined bugged run
            _write_run(root, "run_20261116_00000003", bugged=True, metrics={"val/loss": 2.5})
            # Stuck running > 72h
            _write_run(
                root,
                "run_20261116_00000004",
                status="running",
                created_at="2026-11-10T00:00:00Z",
                metrics={},
            )

            write_markdown_file(
                root / "claims" / "clm_test.md",
                {
                    "title": "Test claim",
                    "status": "supported",
                    "verification": "draft",
                    "goals": ["[[goal_missing]]"],
                    "supports": ["[[run_20261116_00000003]]"],
                    "contradicts": ["[[run_20261116_00000003]]"],
                    "qualifies": [],
                },
                "## Hypothesis & Mechanism\n- TODO: explain\n\n## Evidence Synthesis\n- missing citation\n",
            )

            g = ContourGraph(root)
            now = dt.datetime(2026, 11, 18, 0, 0, tzinfo=dt.timezone.utc)
            issues = find_check_issues(g, now=now)
            codes = {i.code for i in issues}
            for expected in ("C001", "C002", "C003", "C004", "C005", "C007", "C009", "C010"):
                self.assertIn(expected, codes)
            self.assertNotIn("C006", codes)

            c010 = [i for i in issues if i.code == "C010"]
            self.assertEqual(len(c010), 1)
            self.assertEqual(c010[0].severity, WARN)
            c001_cfg = [i for i in issues if i.code == "C001" and "config_path" in i.message]
            self.assertEqual(len(c001_cfg), 1)
            self.assertEqual(c001_cfg[0].severity, ERROR)

    def test_c007_claim_semantics_neutral_standalone_analysis_and_contradicts(self) -> None:
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
            _write_run(
                root,
                "run_20261116_00000002",
                parent="run_20261116_00000001",
                metrics={"val/loss": 2.99},  # clean neutral
            )
            _write_analysis(
                root,
                "ana_20261117_00000001",
                kind="analysis",
                subjects=["run_20261116_00000001"],
                metrics={"svd/eff_rank_layer6": 34.2},  # unregistered diagnostic -> no C009
            )
            _write_run(root, "run_20261116_00000003", metrics={"val/loss": 2.4})  # standalone clean run

            write_markdown_file(
                root / "claims" / "clm_ok.md",
                {
                    "title": "Invariance + mechanistic rank + threshold",
                    "status": "supported",
                    "verification": "draft",
                    "goals": [],
                    "supports": [
                        "[[run_20261116_00000002]]",
                        "[[ana_20261117_00000001]]",
                        "[[run_20261116_00000003]]",
                    ],
                    "contradicts": [],
                    "qualifies": [],
                },
                "## Hypothesis & Mechanism\nHolds.\n\n## Evidence Synthesis\n"
                "- [[run_20261116_00000002]], [[ana_20261117_00000001]], [[run_20261116_00000003]].\n",
            )
            g = ContourGraph(root)
            self.assertEqual(find_check_issues(g), [])

            # Unresolved contradiction on `supported` triggers C007
            fm, body = parse_markdown_file(root / "claims" / "clm_ok.md")
            fm["contradicts"] = ["[[run_20261116_00000001]]"]
            write_markdown_file(
                root / "claims" / "clm_ok.md",
                fm,
                body + "- Contradicted by [[run_20261116_00000001]].\n",
            )
            g = ContourGraph(root)
            c007 = [i for i in find_check_issues(g) if i.code == "C007"]
            self.assertEqual(len(c007), 1)
            self.assertIn("unresolved `contradicts`", c007[0].message)

    def test_taint_propagation_quarantine_and_c008_staleness(self) -> None:
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
            _write_analysis(
                root,
                "ana_20261117_00000001",
                kind="analysis",
                subjects=["run_20261116_00000001"],
                metrics={"svd/rank": 32.0},
            )
            write_markdown_file(
                root / "claims" / "clm_mech.md",
                {
                    "title": "Mechanistic rank claim",
                    "status": "supported",
                    "verification": "draft",
                    "goals": [],
                    "supports": ["[[ana_20261117_00000001]]"],
                    "contradicts": [],
                    "qualifies": [],
                },
                "## Hypothesis & Mechanism\nLow rank.\n\n## Evidence Synthesis\n- [[ana_20261117_00000001]].\n",
            )

            g = ContourGraph(root)
            self.assertEqual(issues_blocking_verify(g, "clm_mech"), [])
            main(["--root", str(root), "verify", "clm_mech", "--agent", "--note", "Clean probe"])
            g = ContourGraph(root)
            self.assertEqual(effective_verification(g, g.claims["clm_mech"]), "agent_audited")

            # Quarantine subject run with `bugged: true` -> propagates taint to analysis & claim
            fm_r, body_r = parse_markdown_file(root / "runs" / "run_20261116_00000001.md")
            fm_r["bugged"] = True
            write_markdown_file(root / "runs" / "run_20261116_00000001.md", fm_r, body_r)

            g = ContourGraph(root)
            self.assertEqual(effective_verification(g, g.claims["clm_mech"]), "draft")
            codes = {i.code for i in find_check_issues(g)}
            self.assertIn("C002", codes)
            self.assertIn("C007", codes)
            self.assertIn("C008", codes)
            rev = find_review_items(g)
            self.assertEqual(len(rev["stale_verified"]), 1)

            # Once unlinked from Claim and analysis quarantined (`bugged: true`), keeping files on disk passes cleanly!
            fm_a, body_a = parse_markdown_file(root / "analyses" / "ana_20261117_00000001.md")
            fm_a["bugged"] = True
            write_markdown_file(root / "analyses" / "ana_20261117_00000001.md", fm_a, body_a)
            fm_c, body_c = parse_markdown_file(root / "claims" / "clm_mech.md")
            fm_c["status"] = "active"
            fm_c["verification"] = "draft"
            fm_c["supports"] = []
            write_markdown_file(root / "claims" / "clm_mech.md", fm_c, body_c)
            g = ContourGraph(root)
            self.assertEqual(find_check_issues(g), [])

    def test_frontier_composes_attached_eval_and_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            (root / "contour.yaml").write_text(yaml.safe_dump({
                "contour": 1,
                "metrics": {
                    "in1k/top1": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.2},
                    "ade20k/miou": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.5},
                    "val/loss": {"direction": "min", "kind": "quality"},  # missing threshold -> C009
                },
            }))
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
            _write_analysis(
                root, "ana_20261117_0000000e",
                kind="eval",
                subjects=["run_20261116_0000000b"],
                metrics={"ade20k/miou": 45.2},
            )
            _write_run(
                root, "run_20261116_0000000c",
                compute={"gpu_hours": 2.0, "params": 1000},
                code={"commit": "c1", "dirty": True},
                metrics={"in1k/top1": 81.0},
            )

            g = ContourGraph(root)
            codes = {i.code for i in find_check_issues(g)}
            self.assertIn("C009", codes)

            _, _, _, pareto_in1k = compute_frontier(g, metric_filter="in1k/top1")
            self.assertEqual(pareto_in1k, {"run_20261116_0000000c"})

            _, _, ade_runs, pareto_ade = compute_frontier(g, metric_filter="ade20k/miou")
            self.assertEqual([r.id for r in ade_runs], ["run_20261116_0000000b"])
            self.assertEqual(pareto_ade, {"run_20261116_0000000b"})

            rev = find_review_items(g)
            dirty_ids = {item["id"] for item in rev["dirty_frontier"]}
            self.assertIn("run_20261116_0000000c", dirty_ids)


if __name__ == "__main__":
    unittest.main()

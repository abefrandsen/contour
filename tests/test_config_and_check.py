import datetime as dt
import tempfile
from pathlib import Path
import unittest

import yaml

from contour.check import (
    ERROR,
    RULE_MISSING_ARTIFACT,
    RULE_OVERCLAIMED_STATUS,
    RULE_SCHEMA_ERROR,
    RULE_STALE_VERIFICATION,
    RULE_STUCK_JOB,
    RULE_TAINTED_EVIDENCE,
    RULE_TODO_PLACEHOLDER,
    RULE_UNDECLARED_METRIC,
    RULE_UNSYNTHESIZED_EVIDENCE,
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
    script_tracked: bool = True,
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
        "script": "scripts/probe.py" if script_tracked else "analyses/scripts/probe.py",
        "script_tracked": script_tracked,
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
                "val/loss": {"direction": "min", "kind": "quality"},
                "sys/tput": {"direction": "max", "kind": "cost", "min_rel_delta": 0.05},
                "attn/entropy": {"direction": "max", "kind": "diagnostic"},
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

    def test_named_check_rules_and_sidecar_validation(self) -> None:
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
            _write_run(root, "run_20261116_00000003", bugged=True, metrics={"val/loss": 2.5})
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
            rules = {i.code for i in issues}
            for expected in (
                RULE_SCHEMA_ERROR,
                RULE_TAINTED_EVIDENCE,
                RULE_UNSYNTHESIZED_EVIDENCE,
                RULE_TODO_PLACEHOLDER,
                RULE_STUCK_JOB,
                RULE_OVERCLAIMED_STATUS,
                RULE_UNDECLARED_METRIC,
                RULE_MISSING_ARTIFACT,
            ):
                self.assertIn(expected, rules)

            art_issues = [i for i in issues if i.code == RULE_MISSING_ARTIFACT]
            self.assertEqual(len(art_issues), 1)
            self.assertEqual(art_issues[0].severity, WARN)
            cfg_issues = [i for i in issues if i.code == RULE_SCHEMA_ERROR and "config_path" in i.message]
            self.assertEqual(len(cfg_issues), 1)
            self.assertEqual(cfg_issues[0].severity, ERROR)

    def test_claim_semantics_untracked_script_downgrade_and_contradicts(self) -> None:
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
            _write_run(root, "run_20261116_00000002", parent="run_20261116_00000001", metrics={"val/loss": 2.99})
            _write_analysis(
                root,
                "ana_20261117_00000001",
                kind="analysis",
                subjects=["run_20261116_00000001"],
                script_tracked=True,
                metrics={"svd/eff_rank_layer6": 34.2},
            )
            _write_analysis(
                root,
                "ana_20261117_00000002",
                kind="analysis",
                subjects=["run_20261116_00000001"],
                script_tracked=False,  # untracked script -> exploratory!
                metrics={"svd/adhoc": 10.0},
            )
            _write_run(root, "run_20261116_00000003", metrics={"val/loss": 2.4})

            # Tracked analysis + clean neutral + standalone run pass in supports; untracked analysis ok in qualifies!
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
                    "qualifies": ["[[ana_20261117_00000002]]"],
                },
                "## Hypothesis & Mechanism\nHolds.\n\n## Evidence Synthesis\n"
                "- [[run_20261116_00000002]], [[ana_20261117_00000001]], [[run_20261116_00000003]], [[ana_20261117_00000002]].\n",
            )
            g = ContourGraph(root)
            self.assertEqual(find_check_issues(g), [])

            # Putting the untracked analysis in `supports` triggers `overclaimed-status`!
            fm, body = parse_markdown_file(root / "claims" / "clm_ok.md")
            fm["supports"].append("[[ana_20261117_00000002]]")
            fm["qualifies"] = []
            write_markdown_file(root / "claims" / "clm_ok.md", fm, body)
            g = ContourGraph(root)
            oc = [i for i in find_check_issues(g) if i.code == RULE_OVERCLAIMED_STATUS]
            self.assertEqual(len(oc), 1)
            self.assertIn("exploratory", oc[0].message)

    def test_verified_hash_covers_claim_body_and_subject_taint(self) -> None:
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
                script_tracked=True,
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
                "## Hypothesis & Mechanism\nLow rank.\n\n## Evidence Synthesis\n- **Holds:** [[ana_20261117_00000001]].\n",
            )

            g = ContourGraph(root)
            self.assertEqual(issues_blocking_verify(g, "clm_mech"), [])
            main(["--root", str(root), "verify", "clm_mech", "--human", "--note", "Clean probe"])
            g = ContourGraph(root)
            self.assertEqual(effective_verification(g, g.claims["clm_mech"]), "human_verified")

            # Editing the Claim's Markdown body prose MUST invalidate `verified_hash`!
            fm_c, body_c = parse_markdown_file(root / "claims" / "clm_mech.md")
            write_markdown_file(
                root / "claims" / "clm_mech.md",
                fm_c,
                body_c.replace("Low rank.", "Works universally across all models!"),
            )
            g = ContourGraph(root)
            self.assertEqual(effective_verification(g, g.claims["clm_mech"]), "draft")
            self.assertIn(RULE_STALE_VERIFICATION, {i.code for i in find_check_issues(g)})

            # Re-verify, then quarantine subject run -> propagates taint & invalidates hash
            main(["--root", str(root), "verify", "clm_mech", "--agent"])
            fm_r, body_r = parse_markdown_file(root / "runs" / "run_20261116_00000001.md")
            fm_r["bugged"] = True
            write_markdown_file(root / "runs" / "run_20261116_00000001.md", fm_r, body_r)

            g = ContourGraph(root)
            rules = {i.code for i in find_check_issues(g)}
            self.assertIn(RULE_TAINTED_EVIDENCE, rules)
            self.assertIn(RULE_OVERCLAIMED_STATUS, rules)
            self.assertIn(RULE_STALE_VERIFICATION, rules)
            rev = find_review_items(g)
            self.assertEqual(len(rev["stale_verified"]), 1)

    def test_frontier_composes_attached_eval_and_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            (root / "contour.yaml").write_text(yaml.safe_dump({
                "contour": 1,
                "metrics": {
                    "in1k/top1": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.2},
                    "ade20k/miou": {"direction": "max", "kind": "quality", "primary": True, "min_delta": 0.5},
                    "val/loss": {"direction": "min", "kind": "quality"},
                },
            }))
            _write_run(root, "run_20261116_0000000a", compute={"gpu_hours": 20.0}, metrics={"in1k/top1": 80.0})
            _write_run(root, "run_20261116_0000000b", compute={"gpu_hours": 5.0}, metrics={"in1k/top1": 80.0})
            _write_analysis(
                root, "ana_20261117_0000000e",
                kind="eval",
                subjects=["run_20261116_0000000b"],
                metrics={"ade20k/miou": 45.2},
            )
            _write_run(
                root, "run_20261116_0000000c",
                compute={"gpu_hours": 2.0},
                code={"commit": "c1", "dirty": True},
                metrics={"in1k/top1": 81.0},
            )

            g = ContourGraph(root)
            self.assertIn(RULE_UNDECLARED_METRIC, {i.code for i in find_check_issues(g)})

            _, _, _, pareto_in1k = compute_frontier(g, metric_filter="in1k/top1")
            self.assertEqual(pareto_in1k, {"run_20261116_0000000c"})

            _, _, ade_runs, pareto_ade = compute_frontier(g, metric_filter="ade20k/miou")
            self.assertEqual([r.id for r in ade_runs], ["run_20261116_0000000b"])
            self.assertEqual(pareto_ade, {"run_20261116_0000000b"})

            rev = find_review_items(g)
            self.assertIn("run_20261116_0000000c", {item["id"] for item in rev["dirty_frontier"]})


if __name__ == "__main__":
    unittest.main()

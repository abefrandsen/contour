import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from contour import Analysis, Eval, Run
from contour.cli import main
from contour.compare import compute_analysis_control_level, get_fresh_vs_parent
from contour.graph import ContourGraph
from contour.mdio import parse_markdown_file, write_markdown_file
from contour.run_logger import _get_git_info, ingest_record


class TestRunLogger(unittest.TestCase):
    def test_lifecycle_sidecar_config_lean_frontmatter_and_author_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            (root / "contour.yaml").write_text(yaml.safe_dump({
                "contour": 1,
                "metrics": {
                    "val/loss": {"direction": "min", "kind": "quality", "primary": True, "min_delta": 0.05},
                },
            }))

            with Run(
                title="Baseline",
                config={"optim": {"lr": 1e-3, "wd": 0.05}, "data": {"splits": ["pile"]}},
                artifacts=[{"path": "logs/base.log", "kind": "log"}],
                root=root,
                git_cwd=root,
            ) as base:
                self.assertEqual(base.status, "running")
                self.assertTrue(base.path and base.path.exists())
                fm_running, _ = parse_markdown_file(base.path)
                self.assertEqual(fm_running["artifacts"][0]["path"], "logs/base.log")
                base.track(3.0, "loss", context={"subset": "val"})
                base.set_compute_budget(gpu_hours=5.0, params=125_000_000)
                base.add_artifact("s3://bucket/base/ckpt.pt", kind="checkpoint")

            fm_b, _ = parse_markdown_file(base.path)
            self.assertEqual(fm_b["status"], "final")
            self.assertEqual(fm_b["metrics"]["val/loss"], 3.0)
            self.assertEqual(fm_b["config_path"], f"runs/configs/{base.id}.yaml")
            for omitted in ("id", "type", "bugged", "config", "vs_parent"):
                self.assertNotIn(omitted, fm_b)
            self.assertTrue((root / fm_b["config_path"]).exists())

            with Run(
                title="Treatment",
                parent=base.id,
                config={"optim": {"lr": 2e-3, "wd": 0.05}, "data": {"splits": ["pile"]}},
                root=root,
                git_cwd=root,
            ) as treat:
                fm_t, _ = parse_markdown_file(treat.path)
                fm_t["title"] = "Author Edited Title"
                write_markdown_file(treat.path, fm_t, "## Key Takeaway (TL;DR)\n- Author prose kept.\n")
                treat.track(2.7, "loss", context={"subset": "val"})
                treat.set_compute_budget(gpu_hours=5.0, params=125_000_000)

            fm_final, body_final = parse_markdown_file(treat.path)
            self.assertEqual(fm_final["title"], "Author Edited Title")
            self.assertIn("Author prose kept.", body_final)
            self.assertEqual(fm_final["parent"], f"[[{base.id}]]")

            g = ContourGraph(root)
            vs = get_fresh_vs_parent(g, g.runs[treat.id])
            self.assertIsNotNone(vs)
            self.assertEqual(vs["control_level"], "clean")
            self.assertEqual(vs["verdict"], "win")
            self.assertIn("optim.lr", vs["config_diff"])

    def test_dirty_check_excludes_contour_dir_and_tracks_analysis_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
            croot = repo / "contour"
            main(["init", str(croot)])
            (repo / "train.py").write_text("x = 1\n")
            (repo / "scripts").mkdir()
            tracked_script = repo / "scripts" / "eval_ood.py"
            tracked_script.write_text("print('tracked eval')\n")
            subprocess.run(
                ["git", "add", "train.py", "scripts/eval_ood.py", "contour/contour.yaml", "contour/.gitignore"],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

            # Run 1 writes uncommitted files inside contour/runs/
            with Run(title="Run 1", config={"lr": 1e-3}, root=croot, git_cwd=repo) as r1:
                r1.track(80.0, "acc", context={"subset": "val"})
            self.assertFalse(r1.code["dirty"])

            # Run 2 runs right after Run 1 without committing contour/ -> MUST still be code.dirty == False!
            with Run(title="Run 2", parent=r1.id, config={"lr": 2e-3}, root=croot, git_cwd=repo) as r2:
                r2.track(81.0, "acc", context={"subset": "val"})
            self.assertFalse(r2.code["dirty"])

            # Ad-hoc gitignored analysis script -> script_tracked=False -> exploratory!
            adhoc_script = croot / "analyses" / "scripts" / "adhoc.py"
            adhoc_script.write_text("print('adhoc')\n")
            with Analysis(
                title="Adhoc probe",
                subjects=[r1.id],
                script=adhoc_script,
                root=croot,
                git_cwd=repo,
            ) as ana_adhoc:
                ana_adhoc.track(12.3, "rank", context={"subset": "svd"})
            fm_adhoc, _ = parse_markdown_file(ana_adhoc.path)
            self.assertFalse(fm_adhoc["script_tracked"])

            # Tracked clean script -> script_tracked=True -> clean!
            with Eval(
                title="Tracked eval",
                subjects=[r1.id],
                script=tracked_script,
                launch_cmd="python scripts/eval_ood.py --ckpt r1",
                root=croot,
                git_cwd=repo,
            ) as ev_tracked:
                ev_tracked.track(71.5, "acc", context={"subset": "ood"})
            fm_ev, _ = parse_markdown_file(ev_tracked.path)
            self.assertTrue(fm_ev["script_tracked"])

            g = ContourGraph(croot)
            self.assertEqual(compute_analysis_control_level(g, g.analyses[ana_adhoc.id]), "exploratory")
            self.assertEqual(compute_analysis_control_level(g, g.analyses[ev_tracked.id]), "clean")

            # Editing actual source code outside contour/ DOES mark dirty
            (repo / "train.py").write_text("x = 2\n")
            dirty_info = _get_git_info(repo, contour_root=croot)
            self.assertTrue(dirty_info["dirty"])

    def test_exception_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            with self.assertRaises(RuntimeError):
                with Run(title="Crash", root=root) as r:
                    raise RuntimeError("OOM")
            fm, _ = parse_markdown_file(r.path)
            self.assertEqual(fm["status"], "failed")

    def test_rank_zero_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            with patch.dict(os.environ, {"RANK": "1"}):
                with Run(title="Worker 1", root=root) as r:
                    r.track(1.0, "val/loss")
                    self.assertFalse(r.enabled)
                    self.assertIsNone(r.path)
            self.assertEqual(list((root / "runs").glob("*.md")), [])

    def test_to_record_and_ingest_record_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dest"
            main(["init", str(root)])
            r = Run(title="Remote Sim", config={"lr": 1e-3}, git_cwd=tmp)
            r.track(80.0, "val/acc")
            r.close()
            rec = r.to_record()
            path = ingest_record(root, rec)
            self.assertTrue(path.exists())
            fm, _ = parse_markdown_file(path)
            self.assertEqual(fm["metrics"]["val/acc"], 80.0)
            self.assertEqual(fm["config_path"], f"runs/configs/{r.id}.yaml")


if __name__ == "__main__":
    unittest.main()

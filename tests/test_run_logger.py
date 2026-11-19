import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from contour import Analysis, Eval, Run
from contour.cli import main
from contour.compare import get_fresh_vs_parent
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
            # Path = identity; no redundant id/type/bugged/vs_parent/config in YAML
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

    def test_analysis_and_eval_logger_script_hash_and_launch_cmd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            script_file = root / "analyses" / "scripts" / "probe_svd.py"
            script_file.write_text("print('svd probe')\n", encoding="utf-8")

            with Run(title="Base", root=root, git_cwd=root) as base:
                base.track(80.0, "acc", context={"subset": "val"})

            with Analysis(
                title="SVD Spectrum Probe",
                subjects=[base.id],
                script=script_file,
                launch_cmd="python analyses/scripts/probe_svd.py",
                root=root,
                git_cwd=root,
            ) as ana:
                ana.track(38.4, "eff_rank", context={"subset": "svd"})
                ana.add_artifact("analyses/artifacts/svd.png", kind="plot", note="Layer decay")

            fm_a, _ = parse_markdown_file(ana.path)
            self.assertEqual(fm_a["kind"], "analysis")
            self.assertEqual(fm_a["status"], "final")
            self.assertEqual(fm_a["script"], "analyses/scripts/probe_svd.py")
            self.assertTrue(fm_a["script_sha256"] and len(fm_a["script_sha256"]) == 16)
            self.assertEqual(fm_a["launch_cmd"], "python analyses/scripts/probe_svd.py")
            for omitted in ("id", "type", "bugged", "config", "config_path", "compute"):
                self.assertNotIn(omitted, fm_a)
            self.assertEqual(fm_a["metrics"]["svd/eff_rank"], 38.4)

            with Eval(
                title="Full OOD Eval",
                subjects=[base.id],
                script=script_file,
                launch_cmd="python eval.py --ckpt base --split ood",
                root=root,
                git_cwd=root,
            ) as ev:
                ev.track(71.5, "acc", context={"subset": "ood"})

            fm_e, _ = parse_markdown_file(ev.path)
            self.assertEqual(fm_e["kind"], "eval")
            self.assertEqual(fm_e["launch_cmd"], "python eval.py --ckpt base --split ood")

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

    def test_git_dirty_detection_and_gitignored_analysis_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
            main(["init", str(repo)])
            (repo / "train.py").write_text("x = 1\n")
            subprocess.run(["git", "add", "train.py", "contour.yaml", ".gitignore"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

            (repo / "analyses" / "scripts" / "adhoc.py").write_text("print(1)\n")
            (repo / "analyses" / "artifacts" / "plot.png").write_bytes(b"PNG")
            clean_info = _get_git_info(repo)
            self.assertFalse(clean_info["dirty"])
            self.assertTrue(clean_info["commit"])

            (repo / "untracked.py").write_text("y = 2\n")
            dirty_info = _get_git_info(repo)
            self.assertTrue(dirty_info["dirty"])

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

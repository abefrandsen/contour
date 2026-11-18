import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from contour import Run
from contour.cli import main
from contour.mdio import parse_markdown_file, write_markdown_file
from contour.run_logger import _get_git_info, ingest_record


class TestRunLogger(unittest.TestCase):
    def test_lifecycle_and_author_preservation(self) -> None:
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
                task="lm",
                arch_family="gpt",
                arch_scale="125M",
                dataset_train=["pile"],
                config={"lr": 1e-3},
                root=root,
                git_cwd=root,
            ) as base:
                self.assertEqual(base.status, "running")
                self.assertTrue(base.path and base.path.exists())
                base.track(3.0, "loss", context={"subset": "val"})
                base.set_compute_budget(gpu_hours=5.0, params=125_000_000)

            fm_b, _ = parse_markdown_file(base.path)
            self.assertEqual(fm_b["status"], "final")
            self.assertEqual(fm_b["metrics"]["val/loss"], 3.0)

            with Run(
                title="Treatment",
                task="lm",
                arch_family="gpt",
                arch_scale="125M",
                dataset_train=["pile"],
                interventions=["lr=0.002"],
                parent_runs=[base.id],
                config={"lr": 2e-3},
                root=root,
                git_cwd=root,
            ) as treat:
                # Author edits title and body while run is running
                fm_t, _ = parse_markdown_file(treat.path)
                fm_t["title"] = "Author Edited Title"
                write_markdown_file(treat.path, fm_t, "## Key Takeaway (TL;DR)\n- Author prose kept.\n")
                treat.track(2.7, "loss", context={"subset": "val"})
                treat.set_compute_budget(gpu_hours=5.0, params=125_000_000)

            fm_final, body_final = parse_markdown_file(treat.path)
            self.assertEqual(fm_final["title"], "Author Edited Title")
            self.assertIn("Author prose kept.", body_final)
            self.assertEqual(fm_final["vs_parent"]["control_level"], "clean")
            self.assertEqual(fm_final["vs_parent"]["verdict"], "win")

    def test_exception_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", str(root)])
            with self.assertRaises(RuntimeError):
                with Run(title="Crash", root=root) as r:
                    raise RuntimeError("OOM")
            fm, _ = parse_markdown_file(r.path)
            self.assertEqual(fm["status"], "failed")
            self.assertIsNone(fm["vs_parent"])

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

    def test_git_dirty_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
            (repo / "train.py").write_text("x = 1\n")
            subprocess.run(["git", "add", "train.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

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
            r = Run(title="Remote Sim", task="cls", arch_family="vit", arch_scale="22M", git_cwd=tmp)
            r.track(80.0, "val/acc")
            r.close()
            rec = r.to_record()
            path = ingest_record(root, rec)
            self.assertTrue(path.exists())
            fm, _ = parse_markdown_file(path)
            self.assertEqual(fm["metrics"]["val/acc"], 80.0)


if __name__ == "__main__":
    unittest.main()

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from contour.bundle import validate_bundle
from contour.theodolite import Theodolite, _get_git_info


class TestTheodolite(unittest.TestCase):
    def test_lifecycle_and_bundle_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bundles"
            with Theodolite(
                exp_name="test_run",
                task="cls",
                dataset_train="tr",
                dataset_eval="ev",
                arch_family="vit",
                arch_scale="22M",
                interventions=["qk_rmsnorm"],
                bundle_dir=bdir,
                use_aim=False,
            ) as run:
                self.assertEqual(run.status, "running")
                run.track(80.0, "top1", context={"subset": "ev"})
                run.log_telemetry("attn/entropy", 2.1)

            self.assertEqual(run.status, "final")
            bundle_path = bdir / f"bundle_{run.run_id}.json"
            self.assertTrue(bundle_path.exists())
            validated = validate_bundle(json.loads(bundle_path.read_text()))
            self.assertEqual(validated["metrics"]["ev/top1"], 80.0)
            self.assertEqual(validated["telemetry"]["attn/entropy"], 2.1)

    def test_exception_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bundles"
            with self.assertRaises(RuntimeError):
                with Theodolite(
                    exp_name="fail_run",
                    task="cls",
                    dataset_train="tr",
                    dataset_eval="ev",
                    arch_family="vit",
                    arch_scale="22M",
                    bundle_dir=bdir,
                    use_aim=False,
                ) as run:
                    raise RuntimeError("OOM")
            self.assertEqual(run.status, "failed")

    def test_rank_zero_guard(self) -> None:
        old = os.environ.get("RANK")
        os.environ["RANK"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bdir = Path(tmp) / "bundles"
                with Theodolite(
                    exp_name="rank1",
                    task="cls",
                    dataset_train="tr",
                    dataset_eval="ev",
                    arch_family="vit",
                    arch_scale="22M",
                    bundle_dir=bdir,
                    use_aim=False,
                ) as run:
                    run.track(50.0, "top1")
                self.assertFalse(run.enabled)
                self.assertFalse(bdir.exists())
        finally:
            if old is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = old

    def test_git_dirty_without_remote_and_with_untracked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.check_call(["git", "init", "-q"], cwd=repo)
            subprocess.check_call(
                ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "init"],
                cwd=repo,
            )
            self.assertFalse(_get_git_info(repo)["dirty"])
            (repo / "new_config.yaml").write_text("lr: 1e-3\n")
            info = _get_git_info(repo)
            self.assertTrue(info["dirty"])
            self.assertIn("new_config.yaml", info["untracked"])


if __name__ == "__main__":
    unittest.main()

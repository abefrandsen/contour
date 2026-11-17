import math
import unittest

from contour.bundle import BundleError, SCHEMA_ID, fieldlog_id_for, validate_bundle


def _base_bundle(**overrides):
    b = {
        "schema": SCHEMA_ID,
        "run_id": "run_abc123",
        "status": "final",
        "created_at": "2026-11-16T10:00:00Z",
        "finished_at": "2026-11-16T11:00:00Z",
        "facets": {"task": "cls", "arch_family": "vit", "arch_scale": "22M"},
        "data": {"train": ["ds_tr"], "eval": ["ds_ev"]},
        "metrics": {"ds_ev/top1": 80.1},
    }
    b.update(overrides)
    return b


class TestBundle(unittest.TestCase):
    def test_valid_minimal_bundle(self) -> None:
        out = validate_bundle(_base_bundle())
        self.assertEqual(out["facets"]["tier"], "proxy")
        self.assertEqual(out["interventions"], [])

    def test_rejects_unknown_and_missing_fields(self) -> None:
        with self.assertRaises(BundleError) as ctx:
            validate_bundle({"schema": SCHEMA_ID, "bogus": 1})
        msg = str(ctx.exception)
        self.assertIn("run_id: required", msg)
        self.assertIn("<root>.bogus: unknown field", msg)

    def test_rejects_non_finite_metric(self) -> None:
        with self.assertRaises(BundleError) as ctx:
            validate_bundle(_base_bundle(metrics={"loss": math.nan}))
        self.assertIn("must be finite", str(ctx.exception))

    def test_rejects_bad_intervention_tag(self) -> None:
        with self.assertRaises(BundleError) as ctx:
            validate_bundle(_base_bundle(interventions=["Bad Tag With Spaces"]))
        self.assertIn("not an atomic tag", str(ctx.exception))

    def test_final_requires_finished_at_and_metrics(self) -> None:
        with self.assertRaises(BundleError) as ctx:
            validate_bundle(_base_bundle(finished_at=None, metrics={}))
        msg = str(ctx.exception)
        self.assertIn("finished_at: required", msg)
        self.assertIn("metrics: a final run", msg)

    def test_fieldlog_id_distinct(self) -> None:
        a = fieldlog_id_for("local_1", "2026-11-16T10:00:00Z")
        b = fieldlog_id_for("local_2", "2026-11-16T10:00:00Z")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("log_20261116_"))


if __name__ == "__main__":
    unittest.main()

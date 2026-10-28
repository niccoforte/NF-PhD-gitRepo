"""Protect paired comparisons, identity checks and physical masking."""
import importlib.util
import runpy
from pathlib import Path
import unittest
import tempfile
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("loss_review", Path(__file__).with_name("review_loss_study.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class LossReviewTests(unittest.TestCase):
    def test_regional_mean_skill_uses_same_mask_and_known_edges(self):
        regional=runpy.run_path(str(Path(__file__).with_name("review_fracture_regions.py")))
        truth=np.arange(24,dtype=float).reshape(2,2,3,2)+1
        mean=np.zeros((2,3,2));valid=np.ones_like(truth,dtype=bool)
        valid[0,0,0]=False
        nodes=np.ones((2,2,3),bool);events=np.ones((2,3,1),bool)
        result=regional["measure_regions"](np.zeros_like(truth),truth,valid,nodes,events,
            np.array([[0,1]]),mean,"UT",np.array([[0,0],[1,1]]))
        self.assertTrue(all(abs(r["skill_percent"])<1e-10 for r in result))
        exact=regional["measure_regions"](truth,truth,valid,nodes,events,
            np.array([[0,1]]),mean,"UT",np.array([[0,0],[1,1]]))
        self.assertTrue(all(abs(r["skill_percent"]-100)<1e-10 for r in exact))

    def test_identity_requires_unique_matching_order(self):
        review.verify_ids([1, 2], ["1", "2"])
        for values in ([2, 1], [1, 1], [1]):
            with self.assertRaises(ValueError):
                review.verify_ids(values, [1, 2])

    def test_missing_and_absent_values_do_not_affect_physical_error(self):
        truth = np.array([1., np.nan, 9.])
        pred = np.array([3., 100., np.nan])
        self.assertEqual(review.error_metrics(pred, truth, [1, 1, 0]), 2.)
        with self.assertRaises(ValueError):
            review.error_metrics(pred, truth, [1, 1, 1])

    def test_changes_are_paired_with_the_same_seed(self):
        rows = pd.DataFrame([dict(variant=v, seed=s, kind="field", mode="UT", rmse=r)
            for v, s, r in (("control", 42, 2.), ("control", 43, 4.),
                            ("spatial", 42, 1.), ("spatial", 43, 6.))])
        result = review.relative_changes(rows)
        np.testing.assert_allclose(result.rmse_change_percent, [0, 0, -50, 50])

    def test_field_decomposition_preserves_masked_squared_error(self):
        truth = np.zeros((2, 3, 2))
        pred = np.array([[[1., 2.], [2., 3.], [3., 4.]], [[9., 1.], [2., 5.], [3., 9.]]])
        mask = np.ones_like(truth, bool); mask[1, 2, 1] = False
        result = review.field_error_decomposition(pred, truth, mask)
        total = review.error_metrics(pred, truth, mask)**2
        self.assertAlmostEqual(result['mean_nodal_error_rmse']**2+result['centered_field_rmse']**2, total)

    def test_report_renders_all_variants_without_query_scope_errors(self):
        rows = pd.DataFrame([dict(variant=v, seed=s, kind=k, mode=m, rmse=1.,
            rmse_change_percent=0., skill_percent=5.) for v in review.VARIANTS
            for s in (42,43,44) for k,m in review.TASKS])
        extra = pd.DataFrame([dict(variant=v, mode=m, metric=k, value=1.)
            for v in review.VARIANTS for m in ("UT","FT")
            for k in ("local_jump_nrmse_percent","peak_height_percent","peak_location_percent","work_percent")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            review.write_report(path, path, rows, extra, pd.DataFrame(), pd.DataFrame(), ["synthetic"])
            text = (path/"README.md").read_text(encoding="utf-8")
            self.assertIn("![Changes in global", text)
            self.assertIn("## Global performance versus the training mean", text)
            self.assertIn("## Best and worst predictions", text)
            self.assertNotIn("fracture-regions.md",text)
            self.assertNotIn("|\n\n|", text)


if __name__ == "__main__":
    unittest.main()

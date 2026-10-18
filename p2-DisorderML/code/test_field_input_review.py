"""Synthetic report ranking/identity checks, not research-accuracy evidence."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd


spec = importlib.util.spec_from_file_location("input_review", Path(__file__).with_name("review_field_inputs.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class InputReviewTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([
            dict(policy=p, seed=s, kind=k, mode=m, rmse=.5 if p == "mixed" else .8, baseline=1.)
            for p in review.POLICIES for s in (42, 43, 44) for k, m in review.TASKS])

    def test_rank_uses_all_four_outputs_and_excludes_oracle(self):
        rows = self.rows()
        oracle = rows[rows.policy == "true"].assign(policy="true_field_oracle", rmse=0.)
        winner, scores = review.select_winner(pd.concat([rows, oracle]))
        self.assertEqual(winner, "mixed")
        self.assertAlmostEqual(scores.loc[winner].mean(), .25)
        self.assertNotIn("true_field_oracle", scores.index)

    def test_reject_missing_task_duplicate_and_invalid_baseline(self):
        rows = self.rows()
        for bad in (rows.iloc[1:], pd.concat([rows, rows.iloc[:1]]), rows.assign(baseline=0.)):
            with self.assertRaises(ValueError):
                review.select_winner(bad)

    def test_ids_preserve_order_and_uniqueness_not_scalar_type(self):
        review.verify_ids([1, 2], ["1", "2"])
        for values in ([2, 1], [1, 1], [1]):
            with self.assertRaises(ValueError):
                review.verify_ids(values, ["1", "2"])


if __name__ == "__main__":
    unittest.main()

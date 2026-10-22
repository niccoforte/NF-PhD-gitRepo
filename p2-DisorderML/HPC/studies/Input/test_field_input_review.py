"""Synthetic report ranking/identity checks, not research-accuracy evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_submission_section_uses_receipts_and_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(review.followup_section(out), [])
            pd.DataFrame([
                dict(role="preflight", seed=42, job_id=100, source_revision="abc"),
                dict(role="spatial", seed=42, job_id=101, source_revision="abc"),
            ]).to_csv(out / "loss-study-jobs.tsv", sep="\t", index=False)
            (out / "loss-study-status.json").write_text(json.dumps(
                dict(checked_at="2026-10-18", summary="Gate completed; production pending.")), encoding="utf-8")
            section = "\n".join(review.followup_section(out))
            self.assertIn("1 full jobs (101–101)", section)
            self.assertIn("Gate completed; production pending.", section)
            self.assertIn("`abc`", section)

    def test_generated_figure_links_are_absolute_but_other_links_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="report images ") as directory:
            source = "![Test figure](test.png)\n[Evidence](selection.json)"
            expected = f"![Test figure](<{(Path(directory) / 'test.png').resolve().as_posix()}>)\n[Evidence](selection.json)"
            self.assertEqual(review.absolute_figure_links(source, Path(directory)), expected)

    def test_corrector_changes_use_paired_seeds_and_reject_missing_cells(self):
        pairs = review.corrector_changes(self.rows())
        self.assertEqual(len(pairs), 12)
        self.assertTrue((abs(pairs.rmse_change_percent - 60) < 1e-9).all())
        rows = self.rows()
        bad = rows[~((rows.policy == "mixed") & (rows.seed == 42) & (rows.kind == "curve") & (rows["mode"] == "UT"))]
        with self.assertRaises(ValueError):
            review.corrector_changes(bad)

    def test_costs_include_correction_once_and_shared_bank_in_both_pipelines(self):
        jobs = []
        for seed in (42, 43, 44):
            for name, hours in ((f"control-bank-1i-s{seed}", 1),
                                (f"input-mixed-s{seed}", 2),
                                (f"input-correction-s{seed}", 3),
                                (f"input-corrected_mixed-s{seed}", 2)):
                jobs.append(dict(name=name, state="COMPLETED", allocated_tres="cpu=12,gres/gpu=1", elapsed_seconds=hours*3600))
        costs = review.corrector_costs(dict(accounting=jobs), "input", "control")
        self.assertTrue((costs.mixed_total_h == 3).all())
        self.assertTrue((costs.corrected_total_h == 6).all())
        self.assertTrue((costs.corrected_followup_h == 5).all())
        jobs[0]["state"] = "RUNNING"
        with self.assertRaises(ValueError):
            review.corrector_costs(dict(accounting=jobs), "input", "control")


if __name__ == "__main__":
    unittest.main()

"""Regression checks for guidance classification and CI failure diagnostics."""

import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import validate_repo


class GuidanceValidationTests(unittest.TestCase):
    def test_ancestor_readmes_cover_only_their_own_subtree(self):
        script = (
            Path(__file__).resolve().parents[2]
            / "maintain-repo-guidance/scripts/check_guidance.py"
        )
        spec = importlib.util.spec_from_file_location("guidance_checker", script)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        study_readme = "p2-DisorderML/HPC/studies/Loss/README.md"
        candidates = checker.guidance_candidates(
            "p2-DisorderML/HPC/studies/Loss/review_loss_study.py"
        )
        self.assertIn(study_readme, candidates)
        self.assertIn("p2-DisorderML/HPC/studies/README.md", candidates)
        self.assertIn("p2-DisorderML/HPC/AGENTS.md", candidates)
        self.assertNotIn(study_readme, checker.guidance_candidates(
            "p2-DisorderML/HPC/studies/Damage/review_status_study.py"
        ))
        self.assertNotIn(study_readme, checker.guidance_candidates(
            "resources/MLstudy_reports.py"
        ))
        self.assertIn("resources/README.md", checker.guidance_candidates(
            "resources/MLstudy_reports.py"
        ))

    def test_winner_is_guidance_without_exempting_model_code(self):
        script = (
            Path(__file__).resolve().parents[2]
            / "maintain-repo-guidance/scripts/check_guidance.py"
        )
        spec = importlib.util.spec_from_file_location("guidance_checker", script)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        self.assertTrue(checker.is_guidance_path("p2-DisorderML/WINNER.md"))
        self.assertFalse(checker.is_guidance_path("resources/MLmodels.py"))
        self.assertFalse(checker.is_guidance_path("unrelated/WINNER.md"))

    def test_errors_survive_warnings_on_stdout(self):
        completed = subprocess.CompletedProcess(
            [], 1, "WARNING: Instruction chain is near the limit.\n",
            "ERROR: Missing guidance coverage.\nERROR: Missing required file.\n",
        )
        with patch.object(validate_repo, "run", return_value=completed):
            result = validate_repo.check_guidance(Path.cwd(), None)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(
            result.detail,
            "ERROR: Missing guidance coverage.\nERROR: Missing required file.",
        )

    def test_successful_guidance_summary(self):
        completed = subprocess.CompletedProcess(
            [], 0, "WARNING: Instruction chain is near the limit.\nGuidance validation passed.\n", ""
        )
        with patch.object(validate_repo, "run", return_value=completed):
            result = validate_repo.check_guidance(Path.cwd(), None)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.detail, "Guidance validation passed.")


if __name__ == "__main__":
    unittest.main()

"""Protect plain-MSE control and isolated FT coefficient semantics."""
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch
import torch

runner = runpy.run_path(str(Path(__file__).with_name("A0-SpatialBalance.py")))
fixture = runpy.run_path(str(Path(__file__).parents[1]/"Locality/test_locality.py"))


class SpatialBalanceTests(unittest.TestCase):
    def test_zero_recovers_masked_mse_and_only_ft_has_spatial_weight(self):
        data = fixture["fixture"]()
        for coefficient in (0., .001, .1):
            args = runner["parse_args"](["--reference-model-json", "unused.json",
                "--run-label", "synthetic", "--spatial-coefficient", str(coefficient)])
            with patch("resources.MLfield.reference_field_edges", side_effect=fixture["edges"]):
                objective = runner["configure"](None, data, args)
            self.assertEqual(objective.field_losses["UT"].spatial_weight, 0)
            loss = objective.field_losses["FT"]
            target = torch.as_tensor(data.splits["val"]["field"]["FT"])
            valid = torch.as_tensor(data.splits["val"]["field_mask"]["FT"])
            pred = (target+torch.randn_like(target)*.2).requires_grad_()
            value = loss(pred, target, mask=valid)
            self.assertEqual(loss.temporal_weight, 0)
            self.assertEqual(loss.localization_gain, 0)
            if coefficient == 0:
                torch.testing.assert_close(value, (pred-target)[valid].square().mean())
            value.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertGreater(float(pred.grad[valid].abs().sum()), 0)
            self.assertEqual(float(pred.grad[~valid].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

"""CNN/spatial combinations change one factor relative to CNN2 controls."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from resources.MLdual import DualStageTransformer

BASE=runpy.run_path(str(Path(__file__).with_name("test_locality.py")))
RUN=runpy.run_path(str(Path(__file__).with_name("A0-CNNSpatial.py")))


class CNNSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_task_weights_masks_gradients_and_checkpoint_config(self):
        for mode in ("UT","FT"):
            data=BASE["fixture"]();net=BASE["network"](data);region=np.array([1,0,1,0,1,1],bool)
            with patch("resources.MLfield.reference_field_edges",side_effect=BASE["edges"]),patch("resources.MLfield.fcc_initial_crack_region",return_value=(region,None)):
                loss=RUN["configure"](net,data,SimpleNamespace(mode=mode,variant="cnn2_spatial"))
            active=loss.field_losses[mode];other=loss.field_losses["FT" if mode=="UT" else "UT"]
            self.assertEqual(active.spatial_weight,.1 if mode=="UT" else .001)
            self.assertEqual((active.temporal_weight,active.localization_gain,other.spatial_weight),(0.,0.,0.))
            if mode=="FT":torch.testing.assert_close(active.spatial_node_mask,torch.as_tensor(region))
            target=torch.as_tensor(data.splits["val"]["field"][mode]);mask=torch.as_tensor(data.splits["val"]["field_mask"][mode])
            pred=(target+.2*torch.randn_like(target)).requires_grad_();active(pred,target,mask).backward()
            self.assertTrue(torch.isfinite(pred.grad).all());self.assertEqual(float(pred.grad[~mask].abs().sum()),0.)
            self.assertEqual(len(net.field_model.tasks[mode].local_refinement.blocks),2)
            clone=DualStageTransformer.from_config(net.get_config());clone.load_state_dict(net.state_dict(),strict=True)


if __name__=="__main__":unittest.main()

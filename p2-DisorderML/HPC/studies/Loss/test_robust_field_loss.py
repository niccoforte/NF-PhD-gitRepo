"""Mask/gradient, serialization and independent field-trainer contracts."""
from pathlib import Path
import runpy
import tempfile
import unittest
import numpy as np
import torch
from resources.MLfield import MaskedRobustFieldLoss
from resources.MLmodels import _model_build_loss_from_config,_model_loss_to_config
from resources.MLdual import DUAL_MODEL


class RobustFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_formula_masks_and_gradients(self):
        for kind in ("mae","smooth_l1"):
            for beta in (.1,1.):
                p=torch.tensor([[[.05,2.,100.]]],requires_grad=True)
                y=torch.tensor([[[0.,0.,float("nan")]]])
                fn=MaskedRobustFieldLoss(kind,beta)
                expected=p[...,:2].abs().mean() if kind=="mae" else torch.nn.functional.smooth_l1_loss(p[...,:2],torch.zeros_like(p[...,:2]),beta=beta)
                torch.testing.assert_close(fn(p,y),expected)
                fn(p,y).backward();self.assertEqual(p.grad[0,0,2],0.)
                self.assertTrue(torch.isfinite(p.grad).all())
                restored=_model_build_loss_from_config(_model_loss_to_config(fn))
                self.assertEqual(type(restored),type(fn));self.assertEqual(restored.get_config(),fn.get_config())

    def test_unknown_and_nonfinite(self):
        p=torch.ones(1,2,3,requires_grad=True);y=torch.full_like(p,float("nan"))
        out=MaskedRobustFieldLoss()(p,y);out.backward()
        self.assertEqual(out,0.);self.assertEqual(p.grad.abs().sum(),0.)
        p=torch.full_like(p,float("inf"));y=torch.zeros_like(p)
        with self.assertRaises(FloatingPointError):MaskedRobustFieldLoss()(p,y)
        self.assertEqual(MaskedRobustFieldLoss()(p,y,torch.zeros_like(y,dtype=torch.bool)),0.)

    def test_independent_field_train_and_reload(self):
        helper=runpy.run_path(str(Path(__file__).parents[1]/"Locality/test_locality.py"))
        runner=runpy.run_path(str(Path(__file__).with_name("A0-RobustFieldLoss.py")))
        from types import SimpleNamespace
        for variant in runner["VARIANTS"]:
            data=helper["fixture"]();net=helper["network"](data)
            loss=runner["configure"](net,data,SimpleNamespace(variant=variant))
            model=DUAL_MODEL(net,loss,data=data,device="cpu",batch=2,field_only_modes=("FT",))
            before={k:v.clone() for k,v in net.curve_model.state_dict().items()}
            model.train(1,verbose=0)
            for k,v in before.items():torch.testing.assert_close(v,net.curve_model.state_dict()[k],rtol=0,atol=0)
            with tempfile.TemporaryDirectory() as td:
                from resources.MLmetrics import postprocess_load_dual_run
                model.save(td);model.save_results(eval_split="val")
                restored=postprocess_load_dual_run(td,load_model=True,device="cpu")[-1]
                self.assertEqual(restored.field_only_modes,("FT",))
                batch=next(iter(model.dataloaders["val"]))
                model.model.eval();restored.model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model._predict_batch(batch)["field"]["FT"],restored._predict_batch(batch)["field"]["FT"])


if __name__=="__main__":unittest.main()

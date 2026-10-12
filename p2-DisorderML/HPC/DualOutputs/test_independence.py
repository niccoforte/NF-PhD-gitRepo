"""CPU isolation, cross-fitting and checkpoint contracts for the eight cases."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest
import subprocess
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import torch

from resources.MLdual import DUAL_DATA, DUAL_MODEL, DualLoss, DualStageTransformer
from resources.MLindependence import CASES, training_groups, prepare_bank, train_case, fold_data, _fit_groups


def fixture():
    rng = np.random.default_rng(7)
    masks = {"UT": np.ones(6, bool), "FT": np.array([1, 0, 1, 1, 1, 1], bool)}
    raw, ids = {}, {}
    for s, n in (("train", 12), ("val", 4), ("test", 3)):
        raw[s] = {"geometry": rng.normal(size=(n, 6, 2)),
                  "field": {m: rng.normal(size=(n, 6, 6)) for m in masks},
                  "field_mask": {m: np.broadcast_to(v[None, :, None], (n, 6, 6)).copy() for m, v in masks.items()},
                  "curve": {m: rng.normal(size=(n, 201)) for m in masks}}
        ids[s] = [f"{s}-{i}" for i in range(n)]
    return DUAL_DATA(raw, masks, {m: np.zeros((6, 4)) for m in masks}, sample_ids=ids,
        metadata={"canonical_coords": np.column_stack((np.arange(6), np.zeros(6))),
                  "field_components": {m: ["U1", "U2"] for m in masks},
                  "field_frame_values": {m: [0., .5, 1.] for m in masks},
                  "curve_x_values": {m: np.linspace(0, 1, 201) for m in masks}})


def model_for(data, case):
    fields, curves, _ = CASES[case]
    small = dict(d_model=8, n_heads=2, n_layers=1, dropout=0., ff_mult=2)
    net = DualStageTransformer.from_data(data, field_kwargs=dict(small, independent_tasks=fields),
        curve_kwargs=dict(small, independent_tasks=curves))
    return DUAL_MODEL(net, DualLoss(), data=data, batch=4, device="cpu")


def arguments():
    return SimpleNamespace(seed=42, batch=4, lr=.001, optimizer="adamw", weight_decay=.001,
        curve_lr_factor=1., grad_clip=None, num_workers=0, epochs=1, early_stop_patience=2,
        early_stop_delta=0., verbose=0, scheduler_factor=.5, scheduler_patience=2,
        scheduler_threshold=.0001, oof_folds=3)


class IndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_eight_group_counts(self):
        self.assertEqual([len(training_groups(c)) for c in CASES], [4, 2, 3, 1, 3, 1, 2, 1])

    def test_no_cross_task_parameters_and_reload(self):
        data = fixture()
        for case, (f, c, _) in CASES.items():
            model = model_for(data, case)
            for independent, stage in ((f, model.model.field_model), (c, model.model.curve_model)):
                if independent:
                    a = {id(p) for p in stage.tasks["UT"].parameters()}
                    b = {id(p) for p in stage.tasks["FT"].parameters()}
                    self.assertFalse(a & b)
            clone = DualStageTransformer.from_config(model.model.get_config())
            clone.load_state_dict(model.model.state_dict(), strict=True)
            batch = next(iter(model.dataloaders["val"]))
            model.model.eval(); clone.eval()
            a = model.model(batch["geometry"], batch["task_features"], batch["node_mask"])
            b = clone(batch["geometry"], batch["task_features"], batch["node_mask"])
            for k in a:
                for m in a[k]:
                    torch.testing.assert_close(a[k][m], b[k][m])
            if f and c:
                a["curve"]["UT"].sum().backward()
                for stage in (model.model.field_model, model.model.curve_model):
                    self.assertTrue(all(p.grad is None for p in stage.tasks["FT"].parameters()))

    def test_fold_normalizers_exclude_heldout(self):
        data = fixture()
        fold = fold_data(data, np.arange(6), np.arange(6, 8), np.arange(8, 12))
        g = data.normalizers["geometry"]
        physical = data.splits["train"]["geometry"][:6]*g["scale"]+g["mean"]
        np.testing.assert_allclose(fold.normalizers["geometry"]["mean"], physical.mean((0, 1), keepdims=True), atol=1e-6)
        self.assertFalse(set(fold.sample_ids["train"]) & set(fold.sample_ids["test"]))
        self.assertFalse(set(fold.sample_ids["val"]) & set(fold.sample_ids["test"]))

    def test_bank_and_all_eight_fits(self):
        data, args = fixture(), arguments()
        with tempfile.TemporaryDirectory() as directory:
            banks = {}
            for case in ("1i", "3i"):
                path = Path(directory)/case; path.mkdir()
                model = model_for(data, case)
                prepare_bank(model, case, args, path, {})
                model.save(path)
                manifest = json.loads((path/"field_bank.json").read_text())
                seen = sum([f["heldout_train_ids"] for f in manifest["folds"]], [])
                self.assertEqual(sorted(seen), sorted(data.sample_ids["train"]))
                self.assertFalse(set(seen) & set(data.sample_ids["test"]))
                banks[case] = path
            for case, (independent, _, joint) in CASES.items():
                model = model_for(data, case)
                args.field_bank = str(banks["1i" if independent else "3i"])
                metadata = {}
                train_case(model, case, args, Path(directory)/("fit-"+case), metadata)
                self.assertTrue(np.isfinite(model.predict("val")["prediction"]["curve"]["FT"]).all())
                self.assertEqual(metadata["case"], case+"p")
                if not joint:
                    bank = torch.load(Path(args.field_bank)/"model.mdl", weights_only=False)
                    for key, value in model.model.field_model.state_dict().items():
                        torch.testing.assert_close(value, bank["model_state_dict"]["field_model."+key])
                path = Path(directory)/("fit-"+case)
                model.save(path)
                model.save_results(eval_split="val")
                clone = model_for(data, case)
                clone.load(path/"model.mdl")
                for k, value in model.model.state_dict().items():
                    torch.testing.assert_close(value, clone.model.state_dict()[k])

    def test_cli_rejects_mixed_ablation_and_missing_bank(self):
        parse = runpy.run_path(str(Path(__file__).with_name("A0-HPC-Dual-test.py")))["parse_args"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(["--independence-case", "1i", "--experiment", "baseline"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(["--independence-case", "3ii", "--experiment", "sudden"])
        args = parse(["--independence-case", "3ii", "--experiment", "baseline"])
        self.assertEqual(args.loss, "mse")
        self.assertEqual(args.field_ft_weight, 1.)

    def test_other_stream_unchanged_after_training_update(self):
        data = fixture()
        model = model_for(data, "1ii")
        before = {k: v.clone() for k, v in model.model.state_dict().items() if ".tasks.FT." in k}
        with tempfile.TemporaryDirectory() as path:
            _fit_groups(model.model, data, training_groups("1ii")[:1], arguments(), path, {})
        for k, value in before.items():
            torch.testing.assert_close(value, model.model.state_dict()[k], rtol=0, atol=0)

    def test_preview_is_non_submitting_and_complete(self):
        script = Path(__file__).with_name("B4_Dual-independence.sh")
        result = subprocess.run(["bash", str(script), "preview-only"], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.count("Case "), 24)
        self.assertIn("240h", result.stdout)
        self.assertIn("six reusable field banks", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""Synthetic unit checks; the adjacent A0 script runs real-data training."""

import tempfile
import unittest
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from resources.MLdual import DUAL_DATA, DUAL_MODEL, DualLoss, DualStageTransformer


def _synthetic_split(rng, samples, nodes, field_features, curve_points, ft_node_mask):
    geometry = rng.normal(size=(samples, nodes, 2)).astype(np.float32)
    field = {
        "UT": rng.normal(size=(samples, nodes, field_features)).astype(np.float32),
        "FT": rng.normal(size=(samples, nodes, field_features)).astype(np.float32),
    }
    field_mask = {
        "UT": np.ones_like(field["UT"], dtype=bool),
        "FT": np.broadcast_to(ft_node_mask[None, :, None], field["FT"].shape).copy(),
    }
    field["FT"][:, ~ft_node_mask, :] = 0.0
    curve = {
        "UT": rng.normal(size=(samples, curve_points)).astype(np.float32),
        "FT": rng.normal(size=(samples, curve_points)).astype(np.float32),
    }
    return {
        "geometry": geometry,
        "field": field,
        "field_mask": field_mask,
        "curve": curve,
    }


class DualMLTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.nodes = 6
        self.field_features = 4
        self.curve_points = 201
        self.node_masks = {
            "UT": np.ones(self.nodes, dtype=bool),
            "FT": np.asarray([True, False, True, True, True, True]),
        }
        x = np.linspace(0.0, 1.0, self.nodes, dtype=np.float32)
        self.task_features = {
            "UT": np.column_stack([x, np.zeros_like(x), np.ones_like(x), self.node_masks["UT"]]),
            "FT": np.column_stack([x, np.zeros_like(x), np.ones_like(x), self.node_masks["FT"]]),
        }
        splits = {
            "train": _synthetic_split(
                rng, 6, self.nodes, self.field_features, self.curve_points, self.node_masks["FT"]
            ),
            "val": _synthetic_split(
                rng, 2, self.nodes, self.field_features, self.curve_points, self.node_masks["FT"]
            ),
            "test": _synthetic_split(
                rng, 2, self.nodes, self.field_features, self.curve_points, self.node_masks["FT"]
            ),
        }
        self.data = DUAL_DATA(
            splits,
            self.node_masks,
            self.task_features,
            sample_ids={
                "train": range(6),
                "val": range(6, 8),
                "test": range(8, 10),
            },
            normalize=True,
            task_feature_names=["x0", "y0", "designable", "present"],
            metadata={
                "canonical_coords": np.column_stack([x, np.zeros_like(x)]),
                "field_components": {mode: ["U1", "U2"] for mode in ("UT", "FT")},
                "field_frame_values": {mode: np.asarray([0.5, 1.0]) for mode in ("UT", "FT")},
                "curve_x_values": {mode: np.linspace(0, 1, self.curve_points) for mode in ("UT", "FT")},
            },
        )
        stage_kwargs = {
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "ff_mult": 2,
            "dropout": 0.0,
            "pos_encoding": "none",
        }
        self.model = DualStageTransformer.from_data(
            self.data,
            field_kwargs=stage_kwargs,
            curve_kwargs=stage_kwargs,
        )
        self.assertEqual(
            sum(isinstance(module, nn.TransformerEncoder) for module in self.model.modules()),
            2,
        )

    def test_joint_forward_masks_and_gradient_flow(self):
        batch = next(iter(self.data.make_dataloaders(batch_size=2)["train"]))
        calls = {"field": 0, "curve": 0}

        def count(stage):
            def hook(_module, _inputs, _output):
                calls[stage] += 1

            return hook

        handles = [
            self.model.field_model.encoder.register_forward_hook(count("field")),
            self.model.curve_model.encoder.register_forward_hook(count("curve")),
        ]
        output = self.model(
            batch["geometry"],
            task_features=batch["task_features"],
            node_masks=batch["node_mask"],
        )
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, {"field": 1, "curve": 1})
        self.assertEqual(tuple(output["field"]["UT"].shape), (2, self.nodes, self.field_features))
        self.assertEqual(tuple(output["field"]["FT"].shape), (2, self.nodes, self.field_features))
        self.assertEqual(tuple(output["curve"]["UT"].shape), (2, self.curve_points))
        self.assertEqual(tuple(output["curve"]["FT"].shape), (2, self.curve_points))
        self.assertTrue(torch.equal(output["field"]["FT"][:, 1], torch.zeros_like(output["field"]["FT"][:, 1])))

        objective = DualLoss(curve_loss={"UT": nn.MSELoss(), "FT": nn.MSELoss()})
        loss, _ = objective(
            output,
            {"field": batch["field"], "curve": batch["curve"]},
            field_masks=batch["field_mask"],
        )
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.field_model.encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.curve_model.encoder.parameters()))

    def test_optional_position_encodings(self):
        batch = next(iter(self.data.make_dataloaders(batch_size=2)["val"]))
        for encoding in ("learned", "sinusoidal"):
            with self.subTest(encoding=encoding):
                config = dict(d_model=8, n_heads=2, n_layers=1, pos_encoding=encoding)
                model = DualStageTransformer.from_data(
                    self.data, field_kwargs=config, curve_kwargs=config
                )
                output = model(batch["geometry"], batch["task_features"], batch["node_mask"])
                self.assertTrue(torch.isfinite(output["curve"]["FT"]).all())

    def test_trial_preset_and_mean_pool(self):
        preset = runpy.run_path(str(Path(__file__).with_name("A0-HPC-Dual-trial1.py")))
        runner = runpy.run_path(str(Path(__file__).with_name("A0-HPC-Dual-test.py")))
        args = runner["parse_args"](preset["DEFAULT_ARGS"] + ["--epochs", "1"])
        self.assertEqual((args.field_d_model, args.curve_n_layers, args.epochs), (256, 2, 1))
        model = DualStageTransformer.from_data(
            self.data, field_kwargs=dict(d_model=8, n_heads=2, n_layers=1),
            curve_kwargs=dict(d_model=8, n_heads=2, n_layers=1, pool=args.curve_pool,
                              use_cls_token=args.curve_cls_token),
        )
        self.assertEqual(model.curve_model.pool, "mean")
        self.assertTrue(model.curve_model.use_cls_token)
        batch = next(iter(self.data.make_dataloaders(batch_size=2)["val"]))
        output = model(batch["geometry"], batch["task_features"], batch["node_mask"])
        self.assertEqual(output["curve"]["FT"].shape, (2, 201))

    def test_absent_ft_node_cannot_affect_retained_ft_outputs(self):
        batch = next(iter(self.data.make_dataloaders(batch_size=2)["test"]))
        self.model.eval()
        with torch.no_grad():
            original = self.model(
                batch["geometry"], batch["task_features"], batch["node_mask"]
            )
            changed_geometry = batch["geometry"].clone()
            changed_geometry[:, 1, :] += 1000.0
            changed = self.model(changed_geometry, batch["task_features"], batch["node_mask"])
        retained = batch["node_mask"]["FT"][0]
        torch.testing.assert_close(
            original["field"]["FT"][:, retained],
            changed["field"]["FT"][:, retained],
        )
        torch.testing.assert_close(original["curve"]["FT"], changed["curve"]["FT"])

    def test_one_optimizer_train_predict_and_checkpoint(self):
        wrapper = DUAL_MODEL(
            self.model,
            DualLoss(),
            data=self.data,
            batch=2,
            lr=1e-3,
            device="cpu",
        )
        wrapper.train(1, verbose=0)
        result = wrapper.predict("test")
        self.assertEqual(tuple(result["prediction"]["curve"]["UT"].shape), (2, self.curve_points))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(wrapper.save(directory))
            self.assertTrue(checkpoint.exists())
            self.assertTrue(checkpoint.with_suffix(".json").exists())
            wrapper.load(checkpoint)

    def test_hpc_runner_archivable_artifacts(self):
        runner = Path(__file__).with_name("A0-HPC-Dual-test.py")
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                str(runner), "--allow-cpu", "--epochs", "1", "--batch", "2",
                "--run-root", directory, "--run-label", "dual-fixture", "--loss", "mse",
                "--field-d-model", "8", "--field-n-heads", "2", "--field-n-layers", "1",
                "--curve-d-model", "8", "--curve-n-heads", "2", "--curve-n-layers", "1",
            ]
            with patch("sys.argv", argv), patch.object(DUAL_DATA, "from_files", return_value=self.data):
                runpy.run_path(str(runner), run_name="__main__")
            run_dir = Path(directory) / "MULTI/Dual/Transformer/dual-fixture"
            self.assertTrue((run_dir / "model.mdl").exists())
            self.assertTrue((run_dir / "model_data.json").exists())
            self.assertTrue((run_dir / "loss_history.csv").exists())
            audit = run_dir / "results/input_audit"
            self.assertTrue((audit / "sample-01.md").is_file())
            with np.load(audit / "input_examples.npz", allow_pickle=False) as inputs:
                self.assertEqual(inputs["raw_disorder"].shape, (3, self.nodes, 2))
            metrics = json.loads((run_dir / "results/metrics.json").read_text())
            self.assertEqual(metrics["evaluation_split"], "val")
            self.assertEqual(metrics["diagnostics"]["field"]["FT"]["n_nodes"], self.nodes - 1)
            self.assertEqual(metrics["diagnostics"]["field"]["UT"]["mean_field_baseline_source"], "train_mean_field")
            with np.load(run_dir / "results/predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(
                    saved["UT_val_curve_truth"], self.data.inverse_curve("UT", self.data.splits["val"]["curve"]["UT"])
                )
                self.assertTrue(np.isnan(saved["FT_val_field_truth"][:, 1]).all())
                self.assertFalse(saved["FT_val_field_mask"][:, 1].any())
                np.testing.assert_array_equal(saved["sample_ids"], ["6", "7"])
            # B1 must be able to find the checkpoint for Slurm log collection.
            self.assertEqual(len(list(run_dir.rglob("*.mdl"))), 1)

    def test_fcc_context_and_sample_pin_membership(self):
        from resources.MLdual import dual_node_context

        # Self-contained fixture: samples/ is intentionally untracked.
        coords = np.asarray([(10*x, 10*y) for y in range(20) for x in range(21)] +
                            [(10*x+5, 10*y+5) for y in range(19) for x in range(20)], dtype=float)
        masks = {"UT": np.ones(800, dtype=bool), "FT": ~((coords[:, 1] == 95) & (coords[:, 0] < 118))}
        designable = (coords[:, 0] > 0) & (coords[:, 0] < 200) & (coords[:, 1] > 0) & (coords[:, 1] < 190)
        features, names, spec = dual_node_context(coords, designable, masks)
        splits = {key: _synthetic_split(np.random.default_rng(4), n, 800, 4, 201, masks["FT"])
                  for key, n in (("train", 3), ("val", 1), ("test", 1))}
        for split in splits.values():
            split["geometry"][:] = 0
        pin = np.flatnonzero(np.all(coords == [55, 35], axis=1))[0]
        splits["train"]["geometry"][1, pin, 0] = -.5
        splits["train"]["geometry"][2, pin, 0] = .5
        data = DUAL_DATA(splits, masks, features, normalize=True, task_feature_names=names,
                         metadata={"canonical_coords": coords, "context_spec": spec})
        coords = data.metadata["canonical_coords"]
        self.assertEqual(int(data.node_masks["UT"].sum()), 800)
        self.assertEqual(int(data.node_masks["FT"].sum()), 788)
        self.assertEqual(int(data.task_features["UT"][:, 2].sum()), 722)
        self.assertEqual(int(data.task_features["UT"][:, 4].sum()), 21)
        self.assertEqual(int(data.task_features["UT"][:, 5].sum()), 21)
        node = int(np.flatnonzero(np.all(coords == [55, 35], axis=1))[0])
        self.assertEqual([int(data.train_dataset[i]["task_features"]["FT"][node, 6]) for i in range(3)], [1, 1, 0])
        np.testing.assert_array_equal(data.task_features["UT"][:, :3], data.task_features["FT"][:, :3])
        scaled, _, _ = dual_node_context(coords * 2 + [7, -2], data.task_features["UT"][:, 2], data.node_masks)
        np.testing.assert_allclose(scaled["FT"], data.task_features["FT"], atol=1e-6)
        invalid = coords.copy()
        invalid[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "Reference coordinates"):
            dual_node_context(invalid, data.task_features["UT"][:, 2], data.node_masks)

    def test_physical_curve_loss_keeps_joint_gradients(self):
        from resources.MLfunc import CombinedCurveLoss

        batch = next(iter(self.data.make_dataloaders(batch_size=2)["val"]))
        output = self.model(batch["geometry"], batch["task_features"], batch["node_mask"])
        objective = DualLoss(curve_loss=CombinedCurveLoss(), curve_normalizers=self.data.normalizers["curve"])
        _, terms = objective(output, {"field": batch["field"], "curve": batch["curve"]}, batch["field_mask"])
        for mode in ("UT", "FT"):
            stats = self.data.normalizers["curve"][mode]
            expected = objective.curve_losses[mode](
                output["curve"][mode] * torch.as_tensor(stats["scale"]) + torch.as_tensor(stats["mean"]),
                batch["curve"][mode] * torch.as_tensor(stats["scale"]) + torch.as_tensor(stats["mean"]),
            )
            torch.testing.assert_close(terms["raw"]["curve"][mode], expected)
        sum(terms["raw"]["curve"].values()).backward()
        for stage in (self.model.field_model, self.model.curve_model):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.encoder.parameters()))

    def test_legacy_data_adapter_aligns_ft_by_coordinates(self):
        rng = np.random.default_rng(11)
        sample_ids = pd.Index(range(10))
        columns = [str(index) for index in range(self.nodes * 2)]
        canonical = np.column_stack(
            [np.arange(self.nodes, dtype=np.float32), np.zeros(self.nodes, dtype=np.float32)]
        )
        delta = rng.normal(scale=0.01, size=(len(sample_ids), self.nodes, 2)).astype(np.float32)
        delta[0] = 0.0
        ut_delta_df = pd.DataFrame(delta.reshape(len(sample_ids), -1), index=sample_ids, columns=columns)
        ut_in_df = pd.DataFrame(
            canonical.reshape(1, -1) + ut_delta_df.to_numpy(),
            index=sample_ids,
            columns=columns,
        )

        ft_native = np.asarray([3, 0, 5, 2, 4])
        ft_columns = np.asarray(columns).reshape(-1, 2)[ft_native].reshape(-1).tolist()
        ft_delta_df = ut_delta_df.loc[:, ft_columns]
        ft_reference = canonical[ft_native]
        ft_in_df = pd.DataFrame(
            ft_reference.reshape(1, -1) + ft_delta_df.to_numpy(),
            index=sample_ids,
            columns=ft_columns,
        )

        base = SimpleNamespace(
            mechMode="MULTI",
            input_kind="field",
            output_kind="curve",
            scale=False,
            reduce_dim=False,
            UTmechTest=True,
            FTmechTest=True,
            PATH="synthetic/",
            geom=SimpleNamespace(l=1.0),
            UT_IN_df=ut_in_df,
            UT_dINr_df=ut_delta_df,
            UT_dIN_df=ut_delta_df.iloc[:, :-2],
            FT_IN_df=ft_in_df,
            FT_dINr_df=ft_delta_df,
            FT_dIN_df=ft_delta_df,
        )
        split_ids = {
            "train": sample_ids[:6],
            "val": sample_ids[6:8],
            "test": sample_ids[8:],
        }
        for mode, mode_coords in (("UT", canonical), ("FT", ft_reference)):
            field_order = np.arange(len(mode_coords))[::-1]
            values = rng.normal(
                size=(len(sample_ids), 2, len(mode_coords), 2)
            ).astype(np.float32)
            setattr(base, f"{mode}_field_input_values", values[:, :, field_order, :])
            setattr(base, f"{mode}_field_input_valid_mask", np.ones_like(values[:, :, field_order, :], dtype=bool))
            setattr(base, f"{mode}_field_input_node_coords", mode_coords[field_order])
            setattr(base, f"{mode}_field_input_index", sample_ids)
            setattr(base, f"{mode}_field_input_frame_values", np.asarray([0.5, 1.0]))
            setattr(base, f"{mode}_field_input_components", ["U1", "U2"])
            setattr(
                base,
                f"{mode}_OUT_df",
                pd.DataFrame([np.linspace(0.0, 1.0, self.curve_points)]),
            )
            for split, ids in split_ids.items():
                setattr(base, f"{mode}_{split}_in_df", pd.DataFrame(index=ids))
                setattr(
                    base,
                    f"{mode}_{split}_out_df",
                    pd.DataFrame(
                        rng.normal(size=(len(ids), self.curve_points)),
                        index=ids,
                    ),
                )

        dual = DUAL_DATA.from_data(base, normalize=False, context_profile="shared")
        self.assertEqual(dual.n_nodes, self.nodes)
        self.assertEqual(int(dual.node_masks["FT"].sum()), len(ft_native))
        self.assertFalse(dual.node_masks["FT"][1])
        self.assertFalse(dual.splits["train"]["field_mask"]["FT"][:, 1, :].any())


if __name__ == "__main__":
    unittest.main()

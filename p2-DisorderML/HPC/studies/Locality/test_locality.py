"""Isolated opt-ins, mask invariance, gradients and field-only saved artifacts."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from resources.MLdual import DUAL_DATA, DUAL_MODEL, DualLoss, DualStageTransformer
from resources.MLlocality import configure_field_locality, MaskedGridRefinement, InitialEdgeBias


VARIANTS = ("control", "degree", "graph", "gated_graph", "adjacency", "geometry", "cnn1", "cnn2")


def fixture():
    rng = np.random.default_rng(3)
    coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    present = {"UT": np.ones(6, bool), "FT": np.array([1, 0, 1, 1, 1, 1], bool)}
    splits, ids = {}, {}
    for split, n in (("train", 8), ("val", 4), ("test", 3)):
        splits[split] = dict(geometry=rng.normal(0, .1, (n, 6, 2)),
            field={m: rng.normal(size=(n, 6, 6)) for m in present},
            field_mask={m: np.broadcast_to(v[None, :, None], (n, 6, 6)).copy() for m, v in present.items()},
            curve={m: rng.normal(size=(n, 201)) for m in present})
        ids[split] = [split+str(i) for i in range(n)]
    return DUAL_DATA(splits, present, {m: np.zeros((6, 4)) for m in present}, sample_ids=ids,
        task_feature_names=["x0", "y0", "designable", "present"],
        metadata=dict(canonical_coords=coords, field_components={m: ["U1", "U2"] for m in present},
            field_frame_values={m: [0, .5, 1] for m in present},
            curve_x_values={m: np.linspace(0, 1, 201) for m in present}))


def edges(coords, mode, present=None):
    pairs = np.array([[0, 1], [1, 2], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5]])
    if present is None:
        present = np.ones(6, bool) if mode == "UT" else np.array([1, 0, 1, 1, 1, 1], bool)
    return pairs[np.asarray(present)[pairs].all(1)]


def network(data):
    torch.manual_seed(42)
    config = dict(d_model=8, n_heads=2, n_layers=2, ff_mult=2, dropout=0., independent_tasks=True)
    return DualStageTransformer.from_data(data, field_kwargs=config, curve_kwargs=dict(config, pool="mean"))


class LocalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_variants_preserve_common_parameters_and_roundtrip(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), patch("resources.MLfield.reference_field_edges", side_effect=edges):
                data = fixture(); net = network(data); old = copy.deepcopy(net.state_dict())
                configure_field_locality(net, data, variant)
                for key, value in old.items():
                    updated = net.state_dict()[key]
                    if variant == "degree" and "field_model" in key and "context_projection" in key:
                        updated = updated[:, :-1]
                    torch.testing.assert_close(updated, value, rtol=0, atol=0)
                clone = DualStageTransformer.from_config(net.get_config())
                clone.load_state_dict(net.state_dict(), strict=True)
                batch = next(iter(data.make_dataloaders(2)["val"]))
                net.eval(); clone.eval()
                with torch.no_grad():
                    a = net.field_model(batch["geometry"], batch["task_features"], batch["node_mask"])
                    b = clone.field_model(batch["geometry"], batch["task_features"], batch["node_mask"])
                for mode in ("UT", "FT"):
                    torch.testing.assert_close(a[mode], b[mode], rtol=0, atol=0)

    def test_absent_nodes_cannot_affect_present_outputs(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), patch("resources.MLfield.reference_field_edges", side_effect=edges):
                data = fixture(); net = network(data); configure_field_locality(net, data, variant)
                net.eval(); batch = next(iter(data.make_dataloaders(2)["val"]))
                g, context, mask = batch["geometry"], batch["task_features"], batch["node_mask"]
                other_g, other_context = g.clone(), {m: v.clone() for m, v in context.items()}
                other_g[:, 1] = 1e6; other_context["FT"][:, 1] = -1e6
                with torch.no_grad():
                    a = net.field_model(g, context, mask)["FT"]
                    b = net.field_model(other_g, other_context, mask)["FT"]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertTrue(torch.all(a[:, 1] == 0))

    def test_new_parameters_receive_field_gradients(self):
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant), patch("resources.MLfield.reference_field_edges", side_effect=edges):
                data = fixture(); net = network(data); configure_field_locality(net, data, variant)
                stage = net.field_model.tasks["FT"]
                batch = next(iter(data.make_dataloaders(2)["val"]))
                output = stage(batch["geometry"], batch["task_features"], batch["node_mask"])["FT"]
                (output-batch["field"]["FT"]).square().mean().backward()
                if variant == "degree":
                    grads = [stage.context_projection.weight.grad[:, -1]]
                else:
                    module = stage.local_graph or stage.local_refinement or stage.attention_bias
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads and any(float(g.abs().sum()) > 0 for g in grads))
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_zero_attention_bias_matches_plain_encoder(self):
        data = fixture(); plain = network(data); biased = copy.deepcopy(plain)
        with patch("resources.MLfield.reference_field_edges", side_effect=edges):
            configure_field_locality(biased, data, "adjacency")
        batch = next(iter(data.make_dataloaders(2)["val"]))
        plain.eval(); biased.eval()
        with torch.no_grad():
            for mode in ("UT", "FT"):
                a = plain.field_model(batch["geometry"], batch["task_features"], batch["node_mask"])[mode]
                b = biased.field_model(batch["geometry"], batch["task_features"], batch["node_mask"])[mode]
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_field_only_skips_other_networks_and_exports_no_curves(self):
        from resources.MLmetrics import postprocess_load_dual_run
        data = fixture(); net = network(data)
        model = DUAL_MODEL(net, DualLoss(), data=data, device="cpu", batch=2, field_only_modes=("FT",))
        other = copy.deepcopy(net.field_model.tasks["UT"].state_dict())
        with patch.object(net.curve_model, "forward", side_effect=AssertionError("curve stage executed")), \
             patch.object(net.field_model.tasks["UT"], "forward", side_effect=AssertionError("inactive task executed")):
            model.train(1, verbose=0)
            with tempfile.TemporaryDirectory() as folder:
                model.save(folder)
                result = Path(model.save_results(eval_split="val"))
                with np.load(result/"predictions.npz") as arrays:
                    self.assertIn("FT_val_field_outputs", arrays)
                    self.assertFalse(any("curve" in key or key.startswith("UT_") for key in arrays.files))
                clone = postprocess_load_dual_run(folder, load_model=True)[-1]
                self.assertEqual(clone.field_only_modes, ("FT",))
        for key, value in other.items():
            torch.testing.assert_close(net.field_model.tasks["UT"].state_dict()[key], value, rtol=0, atol=0)

    def test_cnn_rejects_colliding_or_degenerate_reference_grid(self):
        for coords in ([[0, 0], [0, 0], [1, 1]], [[0, 0], [1, 0]]):
            with self.assertRaises(ValueError):
                MaskedGridRefinement(8, coords)

    def test_full_fcc_grid_and_attention_bias_keep_native_topology(self):
        from resources.MLfield import reference_field_edges
        coords = np.array([(10*x, 10*y) for y in range(20) for x in range(21)]
            + [(10*x+5, 10*y+5) for y in range(19) for x in range(20)], float)
        present = {"UT": np.ones(800, bool), "FT": ~((coords[:, 0] < 118) & (abs(coords[:, 1]-95) < 2))}
        graph = {m: reference_field_edges(coords, m, present[m]) for m in present}
        self.assertEqual((len(graph["UT"]), len(graph["FT"])), (2319, 2259))
        cnn = MaskedGridRefinement(8, coords)
        self.assertEqual((cnn.height, cnn.width, len(torch.unique(cnn.sites))), (39, 41, 800))
        bias = InitialEdgeBias(2, coords, graph, [0, 0], [1, 1])
        bias.adjacency.data.fill_(1.)
        matrix = bias(torch.zeros(1, 800, 2), "FT", torch.as_tensor(present["FT"])[None])
        self.assertTrue(torch.isneginf(matrix[:, :, ~present["FT"]]).all())
        finite = matrix[:, :, present["FT"]]
        self.assertEqual(int((finite == 1).sum()), 2*2*2259)
        self.assertTrue((finite == 0).any())  # global non-edge attention remains.

    def test_scope_conflicts_and_cli(self):
        data = fixture()
        with self.assertRaises(ValueError):
            DUAL_MODEL(network(data), DualLoss(), data=data, field_only_modes=("FT",), true_curve_weight=.5)
        parser = runpy.run_path(str(Path(__file__).with_name("A0-Locality.py")))["parse_args"]
        args = parser(["--reference-model-json", "model.json", "--run-label", "test"])
        self.assertEqual((args.epochs, args.nsims, args.mode, args.variant), (450, None, "UT", "control"))

    def test_real_runner_saves_fields_motion_figures_and_reload(self):
        runner = runpy.run_path(str(Path(__file__).with_name("A0-Locality.py")))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = fixture()
            source = DUAL_MODEL(network(data), DualLoss(), data=data, device="cpu", batch=2)
            source.save(root/"control", metadata=dict(curve_training_source="mixed",
                run_config=dict(early_stop_patience=2, early_stop_delta=0.)))
            for variant in ("degree", "geometry", "cnn2"):
                args = runner["parse_args"](["--reference-model-json", str(root/"control/model.json"),
                    "--run-label", variant, "--run-root", str(root/"runs"), "--epochs", "1",
                    "--nsims", "64", "--mode", "FT", "--variant", variant, "--allow-cpu"])
                with patch.dict(runner["run"].__globals__, {"load_data": lambda args: fixture()}), \
                     patch("resources.MLfield.reference_field_edges", side_effect=edges), \
                     patch("resources.MLfield.fcc_initial_crack_region", return_value=(np.ones(6, bool), np.array([[0, 0], [2, 1]]))):
                    model = runner["run"](args)
                output = Path(model.save_dir)
                for relative in ("model.json", "model.mdl", "model_data.json", "results/metrics.json",
                                 "results/FT_val_field_motion.csv", "results/field-examples.png", "results/loss-history.png"):
                    self.assertTrue((output/relative).is_file(), relative)
                self.assertEqual(json.loads((output/"run_metadata.json").read_text())["checkpoint_reload"], "passed")


if __name__ == "__main__":
    unittest.main()

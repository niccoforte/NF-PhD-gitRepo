"""STATUS-only contract tests; mocked ODB checks do not replace live Abaqus tests."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from resources.damage import reduce_damage, reduce_status, validate_status_arrays


def status_fixture():
    """Two parents, variable subdivisions, unknown frames and an absent joint."""
    arrays = dict(status=np.array([[1, 1, 1], [0, np.nan, 1], [np.nan]*3, [0, 1, 0]], np.float32),
                  parent_strut_index=np.array([0, 0, 1]), strut_id=np.arange(2),
                  element_label=np.array([107, 5, 903]), element_instance=np.array(["PART-1-1"]*3),
                  element_connectivity=np.array([[11, 10], [11, 12], [13, 12]]),
                  chain_order=np.array([0, 1, 0]), subdivision_count=np.array([2, 1]),
                  canonical_endpoint=np.array([[0, 1], [1, 2]]),
                  canonical_xy=np.array([[0., 0.], [2., 0.], [3., 0.], [-1., 0.]]),
                  canonical_node_present=np.array([True, True, True, False]), strut_present=np.ones(2, bool),
                  endpoint_node_label=np.array([[10, 12], [12, 13]]),
                  endpoint_instance=np.full((2, 2), "PART-1-1"),
                  endpoint_xy0=np.array([[[0., 0.], [2., 0.]], [[2., 0.], [3., 0.]]]),
                  node_label=np.array([10, 11, 12, 13]), node_xy0=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]),
                  mapping_residual=np.zeros(2), frame_time=np.array([0., .25, .5, .75]),
                  frame_index=np.arange(4), frame_increment=np.array([0, 10, 20, 30]),
                  schema_version=np.array("strut_status_v1"), task=np.array("UT"), simulation_id=np.array(397))
    metadata = dict(schema_version="strut_status_v1", task="UT", simulation_id=397,
                    failure_rule="whole_element_STATUS_zero_v1", carry_failure=False,
                    source_unchanged_sha256=True, status_reactivations=0,
                    source_sha256=dict(inp="0"*64, odb="1"*64), code_sha256=dict(exporter="2"*64, damage_core="3"*64),
                    mapping=dict(expected_native_struts=2, complete_initial_topology_match=True))
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    arrays.update(reduce_status(arrays["status"], arrays["parent_strut_index"]))
    return arrays


class StatusContract(unittest.TestCase):
    def test_missing_is_unknown_without_carry_and_failure_overrides_missing(self):
        arrays = status_fixture()
        np.testing.assert_array_equal(arrays["strut_failed"], [[False, False], [True, False], [False, False], [True, True]])
        np.testing.assert_array_equal(arrays["strut_failure_known"], [[True, True], [True, True], [False, False], [True, True]])
        np.testing.assert_array_equal(arrays["first_failure_frame"], [1, 3])
        np.testing.assert_array_equal(arrays["previous_active_frame"], [0, 1])
        np.testing.assert_array_equal(arrays["strut_status_coverage"][1], [.5, 1.])
        self.assertEqual(validate_status_arrays(arrays)["missing_status"], 4)

    def test_binary_target_matches_existing_full_audit_reducer(self):
        arrays = status_fixture()
        previous = reduce_damage(np.zeros(arrays["status"].shape), np.arange(3), arrays["status"],
                                 arrays["parent_strut_index"], "failure-only")
        for key in ("element_failed", "element_failure_known", "strut_failed", "strut_failure_known",
                    "failed_element_count", "first_failure_frame", "previous_active_frame"):
            np.testing.assert_array_equal(arrays[key], previous[key])

    def test_reactivation_is_rejected_even_across_missing_frames(self):
        with self.assertRaisesRegex(ValueError, "reactivation"):
            reduce_status([[1], [0], [np.nan], [1]], [0])

    def test_nonbinary_infinite_and_invalid_parent_maps_are_rejected(self):
        for status in ([[.99999]], [[np.inf]], [[-1]], [[2]]):
            with self.assertRaisesRegex(ValueError, "binary"):
                reduce_status(status, [0])
        for parents in ([1], [-1], [.5], [[0]], [0, 1]):
            with self.assertRaises(ValueError):
                reduce_status([[1]], parents)

    def test_validator_rejects_corruption_and_forbidden_object_arrays(self):
        mutations = (
            ("strut_failed", lambda a: a["strut_failed"].__setitem__((0, 0), True)),
            ("times", lambda a: a["frame_time"].__setitem__(2, .25)),
            ("identity", lambda a: a["element_label"].__setitem__(1, 107)),
            ("presence", lambda a: a["canonical_node_present"].__setitem__(1, False)),
            ("order", lambda a: a["frame_index"].__setitem__(1, 99)),
            ("pickle", lambda a: a.update(extra=np.array([{}], object))),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                arrays = status_fixture()
                mutate(arrays)
                with self.assertRaises(ValueError):
                    validate_status_arrays(arrays)

    def test_npz_roundtrip_needs_no_sdeg_or_pickle(self):
        arrays = status_fixture()
        self.assertNotIn("sdeg", arrays)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"FIELDd-test.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(validate_status_arrays(saved), validate_status_arrays(arrays))


class StatusExporter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("status_exporter", Path(__file__).with_name("A2_DamageOUTpostProcess.py"))
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)

    def make_odb(self, arrays, mutation=None):
        instance = SimpleNamespace(name="PART-1-1")
        elements = {int(e): tuple(v) for e, v in zip(arrays["element_label"], arrays["element_connectivity"])}
        nodes = {int(n): v for n, v in zip(arrays["node_label"], arrays["node_xy0"])}
        # Failed grips are outside the selected body and must never mark parents.
        elements[999] = (998, 999)
        nodes.update({998: np.array([0., -1.]), 999: np.array([0., -2.])})
        frames = []
        for j, row in enumerate(arrays["status"]):
            values = [SimpleNamespace(instance=instance, elementLabel=int(label), data=float(value), position="WHOLE_ELEMENT")
                      for label, value in zip(arrays["element_label"], row) if np.isfinite(value)]
            values.append(SimpleNamespace(instance=instance, elementLabel=999, data=0., position="WHOLE_ELEMENT"))
            if mutation == "duplicate" and j == 0:
                values.append(values[0])
            if mutation == "position" and j == 0:
                values[0].position = "INTEGRATION_POINT"
            if mutation == "nonbinary" and j == 0:
                values[0].data = .99999
            fields = {"STATUS": SimpleNamespace(values=values)} if mutation != "missing" else {}
            frames.append(SimpleNamespace(frameValue=arrays["frame_time"][j], incrementNumber=10*j, fieldOutputs=fields))
        odb_instance = SimpleNamespace(elements=[SimpleNamespace(label=e, connectivity=c) for e, c in elements.items()],
                                       nodes=[SimpleNamespace(label=n, coordinates=xy) for n, xy in nodes.items()],
                                       elementSets={"BODYSET": SimpleNamespace(elements=[SimpleNamespace(label=int(e)) for e in arrays["element_label"]])})
        odb = SimpleNamespace(rootAssembly=SimpleNamespace(instances={"PART-1-1": odb_instance}),
                              jobData=SimpleNamespace(version="mock-not-Abaqus"),
                              steps={"Step-1": SimpleNamespace(frames=frames)}, close=Mock())
        mesh = dict(elements=elements, nodes=nodes)
        mapping = dict(instance="PART-1-1", selected_set="BODYSET", simulation_id=397,
                       expected_native_struts=2, complete_initial_topology_match=True,
                       profile="synthetic", canonical_order="synthetic", tolerance=2e-5)
        return odb, mesh, mapping

    def run_export(self, folder, mutation=None):
        arrays = status_fixture()
        odb, mesh, mapping = self.make_odb(arrays, mutation)
        inp = Path(folder)/"Ductile-FCC-20-20disNodes-lhs-all-397.inp"
        path = inp.with_suffix(".odb")
        inp.write_text("synthetic INP placeholder")
        path.write_bytes(b"synthetic ODB placeholder")
        output = Path(folder)/("FIELDd-"+inp.stem+".npz")
        raw_keys = {"strut_id", "canonical_endpoint", "endpoint_node_label", "endpoint_instance", "endpoint_xy0",
                    "subdivision_count", "mapping_residual", "element_label", "element_instance", "element_connectivity",
                    "parent_strut_index", "chain_order", "node_label", "node_xy0", "canonical_xy", "canonical_node_present", "strut_present"}
        mapped = {k: copy.deepcopy(v) for k, v in arrays.items() if k in raw_keys}
        source_paths = dict(inp="/archive/"+inp.name, odb="/archive/"+path.name)
        opener = Mock(return_value=odb)
        with patch.dict(sys.modules, {"odbAccess": SimpleNamespace(openOdb=opener)}), \
                patch.object(self.exporter, "fcc_map", return_value=(mapped, mesh, mapping)):
            try:
                result = self.exporter.export_status(path, inp, "UT", output, source_paths=source_paths)
            finally:
                odb.close.assert_called_once()
                opener.assert_called_once_with(str(path.resolve()), readOnly=True)
        return output, result, arrays

    def test_status_only_ignores_grips_and_has_atomic_validated_products(self):
        with tempfile.TemporaryDirectory() as folder:
            output, result, expected = self.run_export(folder)
            self.assertEqual(result["metadata"]["odb_path"], "/archive/Ductile-FCC-20-20disNodes-lhs-all-397.odb")
            self.assertTrue(result["metadata"]["source_unchanged_sha256"])
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["strut_failed"], expected["strut_failed"])
                self.assertEqual(validate_status_arrays(saved)["struts"], 2)
                self.assertNotIn("sdeg", saved)
                self.assertNotIn("node_displacement", saved)
            marker = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(marker["npz_sha256"], self.exporter.fingerprint(output))
            self.assertFalse(any(p.name.startswith(".") for p in Path(folder).iterdir()))

    def test_missing_duplicate_nonbinary_or_wrong_position_do_not_publish(self):
        for mutation in ("missing", "duplicate", "nonbinary", "position"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    self.run_export(folder, mutation)
                self.assertFalse(list(Path(folder).glob("*.npz")))
                self.assertFalse(list(Path(folder).glob("*.json")))

    def test_roundtrip_failure_preserves_existing_products_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"FIELDd-Ductile-FCC-20-20disNodes-lhs-all-397.npz"
            output.write_bytes(b"existing archive product")
            output.with_suffix(".json").write_text("existing marker")
            calls = []

            def validate(arrays):
                calls.append(True)
                if len(calls) == 2:
                    raise ValueError("synthetic roundtrip failure")
                return validate_status_arrays(arrays)

            with patch.object(self.exporter, "validate_status_arrays", side_effect=validate):
                with self.assertRaisesRegex(ValueError, "synthetic roundtrip"):
                    self.run_export(folder)
            self.assertEqual(output.read_bytes(), b"existing archive product")
            self.assertEqual(output.with_suffix(".json").read_text(), "existing marker")
            self.assertFalse(any(p.name.startswith(".") for p in Path(folder).iterdir()))


if __name__ == "__main__":
    unittest.main()

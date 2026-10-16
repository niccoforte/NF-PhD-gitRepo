"""Synthetic bank identity/geometry/missingness checks; no research data or Abaqus."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from resources.damage import reduce_status, validate_status_arrays
from resources.damage_data import (BANKS, build_damage_banks, index_damage_sources,
                                   read_field_layout, validate_damage_bank)


def _stem(task, sample_id):
    return ("Ductile" if task == "UT" else "Fracture")+"-FCC-20-"+(
        "per-1" if sample_id == 0 else f"20disNodes-lhs-all-{sample_id}")


def _source(task, sample_id, frames=3):
    canonical = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.]])
    count = 4 if task == "UT" else 3
    present = np.arange(4) < count
    edges = np.array([[i, i+1] for i in range(count-1)])
    actual = canonical + np.array([0., .01*sample_id])
    status = np.ones((frames, len(edges)), np.float32)
    if sample_id:
        status[1:, 0] = 0
        status[-1, 1] = np.nan
    meta = dict(schema_version="strut_status_v1", task=task, simulation_id=sample_id,
                source_stem=_stem(task, sample_id), failure_rule="whole_element_STATUS_zero_v1",
                carry_failure=False, source_unchanged_sha256=True, status_reactivations=0,
                mapping=dict(expected_native_struts=len(edges), complete_initial_topology_match=True),
                source_sha256=dict(inp="a"*64, odb="b"*64),
                code_sha256=dict(exporter="c"*64, damage_core="d"*64))
    arrays = dict(schema_version=np.array("strut_status_v1"), task=np.array(task),
                  simulation_id=np.array(sample_id), metadata_json=np.array(json.dumps(meta)),
                  status=status, parent_strut_index=np.arange(len(edges)),
                  frame_time=np.arange(frames, dtype=float)*(1+.00001*sample_id),
                  frame_index=np.arange(frames), frame_increment=np.arange(frames)*10,
                  strut_id=np.arange(len(edges)), strut_present=np.ones(len(edges), bool),
                  canonical_xy=canonical, canonical_node_present=present, canonical_endpoint=edges,
                  endpoint_xy0=actual[edges], endpoint_node_label=edges+101,
                  endpoint_instance=np.full(edges.shape, "PART-1-1"),
                  element_label=np.arange(len(edges))+501,
                  element_instance=np.full(len(edges), "PART-1-1"),
                  element_connectivity=edges+101, node_label=np.arange(count)+101,
                  node_xy0=actual[:count], subdivision_count=np.ones(len(edges), int),
                  chain_order=np.zeros(len(edges), int), mapping_residual=np.zeros(len(edges)))
    arrays.update(reduce_status(status, arrays["parent_strut_index"]))
    validate_status_arrays(arrays)
    return arrays


class DamageData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs, self.transfer, self.output = (self.root/name for name in ("MLdata", "transfer", "staging"))
        self.inputs.mkdir()
        self.transfer.mkdir()
        self.orders = {"UT": np.array([2, 0, 3, 1]), "FT": np.array([2, 0, 1])}
        for task, ids in (("UT", [0, 2]), ("FT", [0, 1, 2])):
            for sample_id in ids:
                np.savez_compressed(self.transfer/("FIELDd-"+_stem(task, sample_id)+".npz"),
                                    **_source(task, sample_id))
        for mech, task in BANKS:
            ids = [2, 0] if mech == "MULTI" else ([0, 2] if task == "UT" else [0, 1, 2])
            reference = _source(task, 0)["canonical_xy"][self.orders[task]]
            coordinates = np.array([reference+np.array([0., .01*sample_id]) for sample_id in ids])
            np.savez_compressed(self.inputs/f"{mech}-{task}-disNodes-allFIELD.npz",
                                Y=np.zeros((len(ids), 3, len(reference), 2)),
                                valid_mask=np.ones((len(ids), 3, len(reference), 2), bool),
                                sample_ids=np.array(ids).astype(str),
                                sample_stems=np.array([_stem(task, sample_id) for sample_id in ids]),
                                node_coords=reference, sample_node_coords=coordinates,
                                frame_values=np.arange(3, dtype=float))

    def _replace(self, path, **changed):
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays.update(changed)
        np.savez_compressed(path, **arrays)

    def _build(self, **kwargs):
        return build_damage_banks(self.inputs, self.output, [self.transfer], **kwargs)

    def _manifest(self, extra=()):
        records = []
        for path in self.transfer.glob("*.npz"):
            with np.load(path, allow_pickle=False) as source:
                records.append(dict(output=str(path), task=str(source["task"]),
                                    simulation_id=int(source["simulation_id"]), source_stem=path.stem[7:]))
        records.extend(extra)
        manifest = self.root/"inventory.json"
        manifest.write_text(json.dumps(dict(schema_version="fcc_status_batch_v1", batches=[dict(records=records)])))
        return manifest

    def test_preserves_id_order_native_edges_unknowns_and_per_sample_times(self):
        reports = self._build()
        self.assertEqual(len(reports), 4)
        for mech, task in BANKS:
            path = self.output/f"{mech}-{task}-disNodes-allD.npz"
            with np.load(path, allow_pickle=False) as bank:
                validate_damage_bank(bank)
                expected_ids = ["2", "0"] if mech == "MULTI" else (["0", "2"] if task == "UT" else ["0", "1", "2"])
                self.assertEqual(bank["sample_ids"].tolist(), expected_ids)
                self.assertEqual(bank["D"].shape, (len(expected_ids), 3, 3 if task == "UT" else 2))
                canonical = _source(task, 0)["canonical_xy"]
                np.testing.assert_array_equal(bank["node_coords"][bank["edge_index"]],
                                              canonical[bank["canonical_endpoint"]])
                row = expected_ids.index("2")
                self.assertEqual(bank["D"][row, -1, 0], 1)
                self.assertTrue(bank["valid_mask"][row, -1, 0])
                self.assertEqual(bank["D"][row, -1, 1], 0)
                self.assertFalse(bank["valid_mask"][row, -1, 1])
                self.assertEqual(bank["frame_time"][row, -1], 2.00004)
                np.testing.assert_array_equal(bank["sample_frame_count"], np.full(len(expected_ids), 3))
                self.assertTrue(bank["frame_present"].all())
                np.testing.assert_array_equal(bank["frame_values"], [0, 1, 2])
                self.assertGreater(json.loads(path.with_suffix(".json").read_text())["unknown_parent_frames"], 0)

    def test_does_not_read_large_displacement_or_validity_arrays(self):
        original = np.lib.npyio.NpzFile.__getitem__
        def guarded(source, key):
            if key in {"Y", "valid_mask"}:
                raise AssertionError("Large target member was loaded")
            return original(source, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded):
            result = read_field_layout(self.inputs/"UT-UT-disNodes-allFIELD.npz")
        self.assertEqual(result["provenance"]["Y_shape"], [2, 3, 4, 2])

    def test_missing_input_does_not_publish_any_bank(self):
        (self.transfer/("FIELDd-"+_stem("FT", 2)+".npz")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing required FIELDd"):
            self._build()
        self.assertFalse(self.output.exists())

    def test_duplicate_product_identity_rejected(self):
        nested = self.transfer/"another-job"
        nested.mkdir()
        np.savez_compressed(nested/("FIELDd-"+_stem("UT", 0)+".npz"), **_source("UT", 0))
        with self.assertRaisesRegex(ValueError, "Duplicate FIELDd task/simulation ID"):
            self._build()

    def test_geometry_mismatch_prevents_all_promotion_and_cleans_staging(self):
        path = self.inputs/"FT-FT-disNodes-allFIELD.npz"
        with np.load(path, allow_pickle=False) as source:
            coords = source["sample_node_coords"]
        coords[1, 0, 1] += .1
        self._replace(path, sample_node_coords=coords)
        with self.assertRaisesRegex(ValueError, "initial sample geometry mismatch"):
            self._build()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_source_longer_than_bank_rejected(self):
        path = self.transfer/("FIELDd-"+_stem("FT", 2)+".npz")
        np.savez_compressed(path, **_source("FT", 2, frames=4))
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            self._build()
        self.assertEqual(list(self.output.iterdir()), [])

    def _short_prefix_bank(self):
        for path in self.transfer.glob("*.npz"):
            with np.load(path, allow_pickle=False) as source:
                task, sample_id = str(source["task"]), int(source["simulation_id"])
            frames = 20 if (task, sample_id) == ("UT", 2) else 21
            np.savez_compressed(path, **_source(task, sample_id, frames=frames))
        for mech, task in BANKS:
            path = self.inputs/f"{mech}-{task}-disNodes-allFIELD.npz"
            with np.load(path, allow_pickle=False) as source:
                n, _, nodes, components = source["Y"].shape
            self._replace(path, Y=np.zeros((n, 21, nodes, components)),
                          valid_mask=np.ones((n, 21, nodes, components), bool),
                          frame_values=np.arange(21, dtype=float))
        return self._build()

    def test_twenty_of_twenty_one_preserves_observations_and_masks_only_trailing_frame(self):
        reports = self._short_prefix_bank()
        for report in reports:
            with np.load(report["output"], allow_pickle=False) as bank:
                summary = validate_damage_bank(bank)
                self.assertTrue(np.isfinite(report["maximum_frame_time_offset_from_field_reference"]))
                if str(bank["task"]) == "FT":
                    self.assertEqual(summary["unavailable_sample_frames"], 0)
                    continue
                row = bank["sample_ids"].tolist().index("2")
                original = _source("UT", 2, frames=20)
                self.assertEqual(bank["sample_frame_count"][row], 20)
                np.testing.assert_array_equal(bank["frame_present"][row], np.arange(21) < 20)
                np.testing.assert_array_equal(bank["D"][row, :20], original["strut_failed"])
                np.testing.assert_array_equal(bank["valid_mask"][row, :20], original["strut_failure_known"])
                np.testing.assert_array_equal(bank["frame_time"][row, :20], original["frame_time"])
                self.assertTrue(np.isnan(bank["frame_time"][row, 20]))
                self.assertFalse(bank["valid_mask"][row, 20].any())
                self.assertFalse(bank["D"][row, 20].any())
                self.assertEqual(bank["D"][row, 19, 0], 1)
                self.assertEqual(bank["first_failure_frame"][row, 0], 1)
                self.assertEqual(bank["first_failure_frame"][row, 2], -1)
                self.assertEqual(summary["unavailable_sample_frames"], 1)
                self.assertEqual(summary["unavailable_parent_frames"], 3)
                self.assertEqual(summary["unknown_parent_frames"],
                                 summary["observed_unknown_parent_frames"]+summary["unavailable_parent_frames"])
                self.assertEqual(summary["samples_with_unavailable_trailing_frames"], 1)
                self.assertEqual(report["samples_with_unavailable_trailing_frame_ids"], ["2"])
                self.assertEqual(report["last_observed_failed_struts_per_sample"][row], 1)
                self.assertEqual(report["final_failed_struts_per_sample"][row], 0)
                self.assertEqual(report["final_unknown_struts_per_sample"][row], 3)

    def test_bank_validator_rejects_invalid_presence_padding_and_times(self):
        self._short_prefix_bank()
        with np.load(self.output/"UT-UT-disNodes-allD.npz", allow_pickle=False) as source:
            original = {key: source[key] for key in source.files}
        row = original["sample_ids"].tolist().index("2")
        for key, position, value, message in (
                ("frame_present", (row, 2), False, "frame prefix/count"),
                ("sample_frame_count", row, 0, "frame prefix/count"),
                ("frame_time", (row, 20), 20., "sample frame times"),
                ("frame_time", (row, 19), np.nan, "sample frame times"),
                ("valid_mask", (row, 20, 0), True, "entirely unknown")):
            with self.subTest(key=key, position=position):
                arrays = {name: array.copy() for name, array in original.items()}
                arrays[key][position] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_damage_bank(arrays)

    def test_source_stem_identity_not_inferred_from_position(self):
        path = self.inputs/"UT-UT-disNodes-allFIELD.npz"
        self._replace(path, sample_stems=np.array([_stem("UT", 2), _stem("UT", 0)]))
        with self.assertRaisesRegex(ValueError, "source stem mismatch"):
            self._build()

    def test_manifest_declared_identity_checked(self):
        manifest = self._manifest()
        self.assertEqual(len(index_damage_sources(inventory=manifest)), 5)
        content = json.loads(manifest.read_text())
        content["batches"][0]["records"][0]["simulation_id"] = 8000
        manifest.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "Manifest/FIELDd identity mismatch"):
            index_damage_sources(inventory=manifest)

    def test_missing_products_for_existing_exclusions_do_not_change_membership(self):
        excluded = [dict(task=task, simulation_id=sample_id, source_stem=_stem(task, sample_id),
                         output=str(self.transfer/("FIELDd-"+_stem(task, sample_id)+".npz")))
                    for task, sample_id in (("UT", 1), ("FT", 3))]
        manifest = self._manifest(excluded)
        reports = build_damage_banks(self.inputs, self.output, inventory=manifest)
        for report in reports:
            scope = report["source_scope"]
            self.assertEqual(scope["manifest_record_count"], 7)
            self.assertEqual(scope["indexed_product_count"], 5)
            self.assertEqual(scope["excluded_manifest_record_count"], 2)
            self.assertEqual(scope["excluded_manifest_ids_by_task"], {"UT": [1], "FT": [3]})
            self.assertTrue(all(not record["product_exists"] for record in scope["excluded_manifest_records"]))
            self.assertIn("not assessed", scope["raw_corpus_completion"])
            with np.load(report["output"], allow_pickle=False) as bank:
                meta = json.loads(str(bank["metadata_json"]))
                self.assertEqual(meta["source_scope"], scope)
                self.assertEqual(bank["sample_ids"].tolist(), report["sample_ids"])

    def test_missing_manifest_product_for_included_id_stops_all_publication(self):
        manifest = self._manifest()
        (self.transfer/("FIELDd-"+_stem("FT", 2)+".npz")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing FIELDd product"):
            build_damage_banks(self.inputs, self.output, inventory=manifest)
        self.assertFalse(self.output.exists())

    def test_existing_bank_is_not_overwritten(self):
        self._build()
        path = self.output/"UT-UT-disNodes-allD.npz"
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "A fracture bank exists"):
            self._build()
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

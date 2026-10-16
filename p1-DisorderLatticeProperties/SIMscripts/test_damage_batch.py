"""Focused FCC batch safety, frame identity, and restart checks without Abaqus."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import A2_DamageBatch as batch
from test_status_contract import status_fixture


class DamageBatch(unittest.TestCase):
    def make_archive(self, folder):
        root = Path(folder) / "archive"
        for identity, job in ((1, "42"), (4501, "43"), (8901, "44"), (0, "6538532")):
            case = root / "disNodes" / "0.2" / "FCC" if identity else root / "per" / "0.0" / "FCC"
            directory = case / job / "zip"
            (directory / "transfer").mkdir(parents=True)
            for token in ("Ductile", "Fracture"):
                stem = token + "-FCC-20-" + ("20disNodes-lhs-all-" + str(identity) if identity else "per-1")
                for suffix in (".inp", ".odb"):
                    (directory / (stem + suffix)).write_bytes(b"synthetic nonempty source")
                (directory / "transfer" / ("FIELDu-" + stem + ".npz")).write_bytes(b"metadata not read by prepare")
        return root

    def make_record(self, folder):
        directory = Path(folder) / "archive" / "disNodes" / "0.2" / "FCC" / "42" / "zip"
        (directory / "transfer").mkdir(parents=True)
        stem = "Ductile-FCC-20-20disNodes-lhs-all-397"
        record = dict(task="UT", simulation_id=397, archive_job_id="42", source_stem=stem,
                      odb=str(directory / (stem + ".odb")), inp=str(directory / (stem + ".inp")),
                      field=str(directory / "transfer" / ("FIELDu-" + stem + ".npz")),
                      output=str(directory / "transfer" / ("FIELDd-" + stem + ".npz")))
        Path(record["odb"]).write_bytes(b"synthetic ODB")
        Path(record["inp"]).write_text("*Instance, name=PART-1-1\n*Node\n10,0,0\n11,1,0\n12,2,0\n13,3,0\n"
                                       "900,10,-1\n901,10,-2\n*Element, type=B21\n107,11,10\n5,11,12\n"
                                       "903,13,12\n999,900,901\n*End Instance\n")
        arrays = status_fixture()
        np.savez_compressed(record["field"], sample_id=np.array(["397"]), mode=np.array(["UT"]),
                            sample_stem=np.array([stem]), node_labels=np.array([12, 900, 10, 13]),
                            node_coords=np.array([[2., 0.], [10., -1.], [0., 0.], [3., 0.]], np.float32),
                            frame_values=arrays["frame_time"] + [0, .0001, .0002, .0003],
                            Y=np.zeros((4, 4, 2)))
        record["source_stat"] = {key: batch.file_stat(record[key]) for key in ("odb", "inp", "field")}
        return record

    def fake_exporter(self, manifest):
        def export(odb, inp, task, output, source_paths):
            arrays = status_fixture()
            metadata = json.loads(str(arrays["metadata_json"]))
            metadata.update(source_sha256=dict(odb=batch.fingerprint(odb), inp=batch.fingerprint(inp)),
                            code_sha256={key: manifest["code_sha256"][key] for key in ("exporter", "damage_core")},
                            odb_path=source_paths["odb"], inp_path=source_paths["inp"])
            arrays["metadata_json"] = np.array(json.dumps(metadata))
            np.savez_compressed(output, **arrays)
        return Mock(side_effect=export)

    def test_preflight_inventory_is_exact_eight_and_never_opens_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_archive(folder)
            with patch.object(Path, "open", side_effect=AssertionError("source read")):
                # Only the maintained code is hashed, so supply that independently.
                with patch.object(batch, "code_fingerprints", return_value={"synthetic": "0"*64}):
                    manifest = batch.prepare_manifest(root, preflight=True)
            self.assertEqual(manifest["records"], 8)
            self.assertEqual(len(manifest["batches"]), 1)
            self.assertEqual({r["simulation_id"] for r in manifest["batches"][0]["records"]}, {0, 1, 4501, 8901})
            self.assertTrue(all(Path(r["output"]).name.startswith("FIELDd-") for r in manifest["batches"][0]["records"]))

    def test_rejects_foreign_lattice_output_redirect_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_archive(folder)
            manifest = batch.prepare_manifest(root, True)
            record = copy.deepcopy(manifest["batches"][0]["records"][0])
            record["output"] = str(Path(folder) / "FIELDd-other.npz")
            with self.assertRaisesRegex(ValueError, "redirected output"):
                batch.validate_record(record, root)
            directory = root / "disNodes" / "0.2" / "FCC" / "42" / "zip"
            (directory / "Ductile-tri-20-per-1.odb").write_bytes(b"wrong lattice")
            with self.assertRaisesRegex(ValueError, "Unexpected ODB"):
                batch.prepare_manifest(root, True)

    def test_manifest_duplicate_and_code_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_archive(folder)
            manifest = batch.prepare_manifest(root, True)
            path = Path(folder) / "manifest.json"
            batch.atomic_json(path, manifest)
            self.assertEqual(batch.load_manifest(path, root)["records"], 8)
            manifest["batches"][0]["records"].append(manifest["batches"][0]["records"][0])
            batch.atomic_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                batch.load_manifest(path, root)
            manifest["code_sha256"]["batch_driver"] = "0"*64
            batch.atomic_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                batch.load_manifest(path, root)

    def test_reviewed_manifest_sha_is_checked_before_source_or_code_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_archive(folder)
            path = Path(folder) / "manifest.json"
            batch.atomic_json(path, batch.prepare_manifest(root, True))
            checksum = batch.fingerprint(path)
            result = batch.run_batch(path, 0, dry_run=True, archive_root=root, manifest_sha256=checksum)
            self.assertEqual(result["state"], "dry_run")
            path.write_text(path.read_text() + "\n")
            with patch.object(batch, "load_manifest", side_effect=AssertionError("must not inspect modified manifest")):
                with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                    batch.run_batch(path, 0, dry_run=True, archive_root=root, manifest_sha256=checksum)

    def test_dry_run_has_no_filesystem_writes_or_abaqus_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_archive(folder)
            path = Path(folder) / "manifest.json"
            batch.atomic_json(path, batch.prepare_manifest(root, True))
            before = {str(p): p.stat().st_mtime_ns for p in Path(folder).rglob("*")}
            with patch.object(batch.importlib.util, "spec_from_file_location", side_effect=AssertionError("Abaqus import")):
                result = batch.run_batch(path, 0, dry_run=True, archive_root=root)
            after = {str(p): p.stat().st_mtime_ns for p in Path(folder).rglob("*")}
            self.assertEqual(result["state"], "dry_run")
            self.assertEqual(before, after)

    def test_frame_index_alignment_uses_exact_labels_and_accepts_small_time_differences(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.make_record(folder)
            result = batch.validate_field_alignment(record, status_fixture())
            self.assertAlmostEqual(result["maximum_time_difference_s"], .0003)
            self.assertFalse(result["interpolation"])
            self.assertEqual(result["field_nodes_outside_selected_body"], 1)
            self.assertEqual(batch.array_header(record["field"], "Y"), (4, 4, 2))
            arrays = status_fixture()
            arrays["endpoint_node_label"][1, 1] = 99
            with self.assertRaisesRegex(ValueError, "endpoints missing"):
                batch.validate_field_alignment(record, arrays)

    def test_publish_and_restart_require_valid_npz_sidecar_and_source_code_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.make_record(folder)
            manifest = dict(code_sha256=batch.code_fingerprints())
            exporter = self.fake_exporter(manifest)
            scratch = Path(folder) / "record"
            result = batch.export_one(record, manifest, scratch, exporter)
            self.assertEqual(result["state"], "produced")
            self.assertFalse(scratch.exists())
            self.assertTrue(batch.reusable_product(record, manifest))
            self.assertEqual(batch.export_one(record, manifest, scratch, exporter)["state"], "skipped_valid")
            self.assertEqual(exporter.call_count, 1)
            sidecar = Path(record["output"]).with_suffix(".json")
            saved = json.loads(sidecar.read_text())
            saved["npz_sha256"] = "0"*64
            sidecar.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "not reusable"):
                batch.export_one(record, manifest, scratch, exporter)
            self.assertFalse(scratch.exists())
            self.assertEqual(batch.export_one(record, manifest, scratch, exporter, replace_invalid=True)["state"], "produced")

    def test_changed_sources_fail_before_copy_or_export(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.make_record(folder)
            Path(record["odb"]).write_bytes(b"modified ODB")
            exporter = Mock(side_effect=AssertionError("must not export"))
            with self.assertRaisesRegex(ValueError, "changed since inventory"):
                batch.export_one(record, dict(code_sha256=batch.code_fingerprints()), Path(folder) / "record", exporter)
            exporter.assert_not_called()

    def test_unpinned_exporter_fingerprint_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.make_record(folder)
            manifest = dict(code_sha256=batch.code_fingerprints())
            original_exporter = self.fake_exporter(manifest)

            def changed_exporter(*args, **kwargs):
                original_exporter(*args, **kwargs)
                output = args[3]
                with np.load(output, allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in saved.files}
                metadata = json.loads(str(arrays["metadata_json"]))
                metadata["code_sha256"]["exporter"] = "0"*64
                arrays["metadata_json"] = np.array(json.dumps(metadata))
                np.savez_compressed(output, **arrays)

            with self.assertRaisesRegex(ValueError, "Exporter/code manifest fingerprint mismatch"):
                batch.export_one(record, manifest, Path(folder) / "record", changed_exporter)
            self.assertFalse(Path(record["output"]).exists())

    def test_failed_export_does_not_publish_and_retains_owned_scratch(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.make_record(folder)
            scratch = Path(folder) / "record"
            with self.assertRaisesRegex(RuntimeError, "mock extraction"):
                batch.export_one(record, dict(code_sha256=batch.code_fingerprints()), scratch,
                                 Mock(side_effect=RuntimeError("mock extraction failure")))
            self.assertTrue(scratch.exists())
            self.assertFalse(Path(record["output"]).exists())
            self.assertFalse(Path(record["output"]).with_suffix(".json").exists())

    def test_batch_failure_is_recorded_and_retains_only_guarded_job_scratch(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.make_record(folder)
            manifest = dict(schema_version=batch.SCHEMA, code_sha256=batch.code_fingerprints(), preflight=False,
                            records=1, batches=[dict(archive_job_id="42", case="disordered", records=[record])])
            path = Path(folder) / "manifest.json"
            batch.atomic_json(path, manifest)
            fake = SimpleNamespace(export_status=Mock(side_effect=RuntimeError("synthetic failure")))
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}), \
                    patch.object(batch.importlib.util, "spec_from_file_location", return_value=spec), \
                    patch.object(batch.importlib.util, "module_from_spec", return_value=fake):
                result = batch.run_batch(path, 0, Path(folder) / "12345", archive_root=Path(folder) / "archive")
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["failed"], 1)
            self.assertTrue((Path(folder) / "12345" / "damage-batch-0" / "record-0").is_dir())
            saved = json.loads((Path(folder) / "results" / "12345-batch-0.json").read_text())
            self.assertIn("synthetic failure", saved["records"][0]["error"])
            self.assertFalse(Path(record["output"]).exists())

    def test_wrapper_damage_dry_run_does_not_load_modules_or_create_directories(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path("/bin/bash")
        if not bash.exists():
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            scripts = repo / "p1-DisorderLatticeProperties" / "SIMscripts"
            scripts.mkdir(parents=True)
            (scripts / "A2_DamageBatch.py").write_text("import sys; assert '--dry-run' in sys.argv; print('SAFE_DRY_RUN')\n")
            wrapper = Path(batch.__file__).with_name("B2_ABAQUS-PPscratch-new.sh")
            environment = dict(os.environ, REPO_ROOT=repo.as_posix(), DRY_RUN="true", DAMAGE_PLAN_PYTHON=os.sys.executable.replace("\\", "/"))
            result = subprocess.run([str(bash), str(wrapper), "--damage-only", "--manifest", "not-opened.json"],
                                    env=environment, cwd=folder, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "SAFE_DRY_RUN")
            self.assertEqual(set(p.name for p in repo.iterdir()), {"p1-DisorderLatticeProperties"})


if __name__ == "__main__":
    unittest.main()

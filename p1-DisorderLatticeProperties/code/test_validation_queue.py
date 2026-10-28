"""Synthetic operational tests; fake solver text is not Abaqus evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from validation_queue import execute_abaqus, run_queue, sha256, validate_manifest, worker_lock, write_json


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="codex_validation_queue_")
        self.base = Path(self.temp.name)
        self.root = self.base / "queue"
        self.manifest = self.base / "manifest.json"
        self.jobs = []
        for name in ("fixture_a", "fixture_b"):
            inp = self.base / (name + ".inp")
            inp.write_text("** Synthetic queue fixture, not a physical model\n")
            self.jobs.append({"job": name, "input": str(inp), "input_sha256": sha256(inp),
                              "cpus": 1, "required_outputs": [".odb", ".dat"]})
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        write_json(self.manifest, {"queue_root": str(self.root), "abaqus": "fake-solver",
                                   "jobs": self.jobs})

    def fake_solver(self, abaqus, job, work, on_started):
        on_started(123, ["fake-solver", job["job"]])
        for suffix in (".odb", ".dat"):
            (work / (job["job"] + suffix)).write_text("Synthetic artifact, no simulation data")
        (work / "launcher.txt").write_text("Abaqus JOB " + job["job"] + " COMPLETED\n")
        return 0

    def test_series_delivery_cleanup_and_idempotent_completion(self):
        order = []
        def execute(*args):
            if order:
                self.assertFalse((self.root / "scratch" / order[-1]).exists())
                self.assertTrue((self.root / "runs" / order[-1] / "delivery.json").exists())
            order.append(args[1]["job"])
            return self.fake_solver(*args)
        state = run_queue(self.manifest, execute, self.base)
        self.assertEqual(order, ["fixture_a", "fixture_b"])
        self.assertEqual(state["phase"], "completed")
        self.assertEqual([r["status"] for r in state["jobs"]], ["completed", "completed"])
        self.assertEqual(list((self.root / "scratch").iterdir()), [])
        # A completed queue never invokes the executor again.
        run_queue(self.manifest, lambda *args: self.fail("Unexpected rerun"), self.base)

    def test_input_tampering_and_duplicate_names_rejected(self):
        Path(self.jobs[0]["input"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed source"):
            validate_manifest(self.manifest, self.base)
        self.jobs[0]["input_sha256"] = sha256(self.jobs[0]["input"])
        self.jobs[1]["job"] = self.jobs[0]["job"]
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(self.manifest, self.base)

    def test_failure_stops_and_retains_scratch_when_odb_missing(self):
        def failed(abaqus, job, work, on_started):
            (work / "launcher.txt").write_text("Abaqus JOB " + job["job"] + " NOT COMPLETED")
            return 1
        state = run_queue(self.manifest, failed, self.base)
        self.assertEqual(state["phase"], "stopped_after_failure")
        self.assertEqual(state["jobs"][1]["status"], "queued")
        self.assertTrue((self.root / "scratch/fixture_a/scratch-owner.json").exists())
        with self.assertRaisesRegex(ValueError, "requires explicit review"):
            run_queue(self.manifest, failed, self.base)

    def test_second_worker_cannot_own_queue(self):
        self.root.mkdir()
        with worker_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another worker"):
                with worker_lock(self.root):
                    self.fail("Duplicate worker lock succeeded")

    def test_separate_queues_share_validation_wide_serial_lock(self):
        audit = self.base / "audit"
        audit.mkdir()
        with worker_lock(audit, "serial-worker.lock"):
            with self.assertRaisesRegex(RuntimeError, "Another worker"):
                run_queue(self.manifest, lambda *args: self.fail("Overlapping solver"), self.base)
        self.assertFalse((self.root / "state.json").exists())
        self.assertEqual(run_queue(self.manifest, self.fake_solver, self.base)["phase"], "completed")

    def test_atomic_state_write_tolerates_brief_windows_reader_lock(self):
        import os
        replace = os.replace
        attempts = []
        def transient_lock(source, target):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("Synthetic brief Windows lock")
            return replace(source, target)
        with mock.patch("validation_queue.os.replace", transient_lock), mock.patch("validation_queue.time.sleep"):
            write_json(self.base / "state.json", {"phase": "completed"})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads((self.base / "state.json").read_text()), {"phase": "completed"})

    def test_started_callback_error_still_waits_for_launched_solver(self):
        process = mock.Mock(pid=456)
        process.wait.return_value = 0
        work = self.base / "callback_fixture"
        work.mkdir()
        def failed_record(*args):
            raise PermissionError("Synthetic persistent state-write failure")
        with mock.patch("validation_queue.subprocess.Popen", return_value=process):
            with self.assertRaises(PermissionError):
                execute_abaqus("fake-solver", self.jobs[0], work, failed_record)
        process.wait.assert_called_once_with()

    def test_crashed_worker_owner_blocks_a_different_queue(self):
        audit = self.base / "audit"
        audit.mkdir()
        write_json(audit / "serial-owner.json", {"phase": "active", "queue_root": "previous-queue"})
        with self.assertRaisesRegex(RuntimeError, "previous validation worker"):
            run_queue(self.manifest, lambda *args: self.fail("Orphan overlap"), self.base)
        self.assertFalse((self.root / "state.json").exists())

    def test_escaped_root_is_rejected(self):
        data = json.loads(self.manifest.read_text())
        data["queue_root"] = str(self.base.parent / "not_owned")
        write_json(self.manifest, data)
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.base)

    def test_linked_subroot_rejected_before_submission(self):
        real_resolve = Path.resolve
        target = self.root / "scratch"
        def resolve(path, *args, **kwargs):
            if path == target:
                return self.base.parent / "escaped_scratch"
            return real_resolve(path, *args, **kwargs)
        with mock.patch.object(Path, "resolve", resolve):
            with self.assertRaises(ValueError):
                validate_manifest(self.manifest, self.base)

    def test_cpus_must_be_integer_not_bool_or_float(self):
        for invalid in (True, 1.5, "8"):
            self.jobs[0]["cpus"] = invalid
            self.save()
            with self.assertRaisesRegex(ValueError, "CPU"):
                validate_manifest(self.manifest, self.base)

    def test_wrong_name_or_empty_required_artifact_blocks_cleanup(self):
        def invalid_output(abaqus, job, work, on_started):
            (work / "unrelated.odb").write_text("Not the requested job")
            (work / (job["job"] + ".dat")).write_text("")
            (work / "launcher.txt").write_text("Abaqus JOB " + job["job"] + " COMPLETED")
            return 0
        state = run_queue(self.manifest, invalid_output, self.base)
        self.assertEqual(state["phase"], "stopped_after_failure")
        self.assertTrue((self.root / "scratch/fixture_a").exists())


if __name__ == "__main__":
    unittest.main()

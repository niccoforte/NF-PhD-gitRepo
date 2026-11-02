"""Continuation ledger regression; no solver or research artifacts are used."""
import json
from pathlib import Path
import tempfile
import unittest

from validation_inventory import queue_ledger


class QueueInventory(unittest.TestCase):
    def test_worker_status_receipt_is_not_read_as_a_second_submission_plan(self):
        with tempfile.TemporaryDirectory(prefix="codex_p1_queue_inventory_") as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"queue_root": str(root / "queue"),
                                            "jobs": [{"job": "fixture"}]}))
            (root / "serial-continuation-fixture.json").write_text(json.dumps({
                "parent_queue": "parent", "following_manifests": [{"path": str(manifest)}]}))
            (root / "serial-continuation-fixture-status.json").write_text(json.dumps({
                "phase": "waiting_for_parent", "worker_pid": 123}))
            result = queue_ledger(root)
            self.assertEqual(result["counts"], {"queued_after_parent": 1})
            self.assertEqual(result["jobs"][0]["job"], "fixture")
            self.assertEqual(len(result["jobs"][0]["occurrences"]), 1)


if __name__ == "__main__":
    unittest.main()

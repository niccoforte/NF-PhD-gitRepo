"""Submission immutability and interrupted-preflight ownership, without Abaqus."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validation_completion as completion
import validation_queue as queue


class CompletionOwnership(unittest.TestCase):
    def test_submitted_prepare_cannot_rewrite_inputs(self):
        with tempfile.TemporaryDirectory(prefix="p1-completion-guard-") as tmp:
            root = Path(tmp)
            out = root / "audit/new-validation"
            (out / "queue-20261029").mkdir(parents=True)
            (out / "queue-20261029/state.json").write_text('{"phase":"running"}')
            source = out / "submitted.inp"
            source.write_bytes(b"immutable submitted input\n")
            for refresh in (False, True):
                with patch.object(completion, "pair_checks", side_effect=AssertionError("must not prepare")):
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        completion.prepare(root, refresh)
            self.assertEqual(source.read_bytes(), b"immutable submitted input\n")

    def test_submitted_preflight_cannot_touch_receipts_or_launch(self):
        with tempfile.TemporaryDirectory(prefix="p1-completion-guard-") as tmp:
            root = Path(tmp)
            state = root / "audit/new-validation/queue-20261029/state.json"
            state.parent.mkdir(parents=True)
            state.write_text('{"phase":"completed"}')
            with patch.object(queue, "validate_manifest", side_effect=AssertionError("must not launch")):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    completion.datacheck(root)

    def test_interrupted_preflight_keeps_durable_owner_active(self):
        with tempfile.TemporaryDirectory(prefix="p1-completion-guard-") as tmp:
            root = Path(tmp)
            out = root / "audit/new-validation"
            out.mkdir(parents=True)
            source = out / "input.inp"
            source.write_bytes(b"fixture input\n")
            owner = root / "audit/serial-owner.json"
            owner.write_text('{"phase":"settled"}')
            manifest = {"abaqus":"fixture-no-execution", "jobs":[{
                "job":"fixture", "input":str(source),
                "input_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}]}
            with patch.object(queue, "validate_manifest", return_value=(manifest, out)), \
                 patch.object(queue, "worker_lock", return_value=nullcontext()), \
                 patch.object(completion.subprocess, "run", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    completion.datacheck(root)
            recorded = json.loads(owner.read_text())
            self.assertEqual(recorded["phase"], "active")
            self.assertEqual(recorded["kind"], "input_datacheck")
            self.assertIn("datacheck", recorded["command"])


if __name__ == "__main__":
    unittest.main()

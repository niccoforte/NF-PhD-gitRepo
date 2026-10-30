"""Submission immutability and interrupted-preflight ownership, without Abaqus."""
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validation_completion as completion
from validation_completion import merge_completion_attempts
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


def receipt(identity, digest, qualified, job="original"):
    return {"case":{"case_id":identity, "source_odb":"C:/fixture/"+job+".odb"},
            "before":{"sha256":digest}, "initiation_prefix_qualified":qualified,
            "state":"qualified" if qualified else "unqualified",
            "event_json":"C:/fixture/"+job+".json", "event_json_sha256":digest+"event",
            "history_export_directory":"C:/fixture/"+job}


class CompletionAttemptSelection(unittest.TestCase):
    def test_preserves_four_qualified_cases_when_adding_a_new_case(self):
        existing = [receipt(str(i), str(i), True) for i in range(4)]
        before = copy.deepcopy(existing)
        added = receipt("hex", "new", True)
        selected, history = merge_completion_attempts(existing, [added])
        self.assertEqual(selected, existing+[added])
        self.assertEqual(existing, before)
        self.assertEqual(history, [])

    def test_unqualified_replacement_cannot_displace_qualified_result(self):
        good, bad = receipt("same", "good", True), receipt("same", "bad", False, "retry")
        selected, history = merge_completion_attempts([good], [bad])
        self.assertEqual(selected, [good])
        self.assertEqual(history[0]["source_sha256"], "bad")

    def test_qualified_restart_replaces_partial_with_old_source_lineage(self):
        partial, restart = receipt("same", "partial", False), receipt("same", "restart", True, "r1")
        selected, history = merge_completion_attempts([partial], [restart])
        self.assertEqual(selected, [restart])
        self.assertEqual(history[0]["source_sha256"], "partial")
        self.assertFalse(history[0]["initiation_prefix_qualified"])

    def test_two_unqualified_attempts_remain_unavailable(self):
        old, retry = receipt("same", "old", False), receipt("same", "retry", False, "r1")
        selected, history = merge_completion_attempts([old], [retry])
        self.assertEqual(selected, [old])
        self.assertFalse(selected[0]["initiation_prefix_qualified"])
        self.assertEqual(history[0]["source_sha256"], "retry")

    def test_reimport_is_idempotent(self):
        row = receipt("same", "same", True)
        selected, history = merge_completion_attempts([row], [copy.deepcopy(row)])
        self.assertEqual(selected, [row])
        self.assertEqual(history, [])

    def test_rejected_attempt_is_not_duplicated_in_history(self):
        good, bad = receipt("same", "good", True), receipt("same", "bad", False, "r1")
        selected, history = merge_completion_attempts([good], [bad])
        selected, repeated = merge_completion_attempts(selected, [bad], history)
        self.assertEqual(selected, [good])
        self.assertEqual(repeated, history)


if __name__ == "__main__":
    unittest.main()

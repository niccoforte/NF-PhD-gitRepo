"""Synthetic completion-gate checks; no research archives or ODBs are opened."""
import copy
import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest

REVIEW = runpy.run_path(str(Path(__file__).with_name("review_damage_dataset.py")))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def evidence():
    memberships = {task: list(range(n)) for task, n in (("UT", 8882), ("FT", 8928), ("MULTI", 8811))}
    rows, transfers, banks = [], [], []
    for task, struts in (("UT", 2319), ("FT", 2259)):
        validation = dict(frames=21, struts=struts, elements=1, known_parent_frames=21*struts,
                          failed_parent_frames=0, final_failed_struts=0, final_failed_elements=0,
                          missing_status=0, status_reactivations=0)
        for i in range(9001):
            rows.append(dict(task=task, simulation_id=i, output=f"/synthetic/{task}/FIELDd-model-{i}.npz",
                             state="produced", npz_sha256=sha(f"{task}-{i}"), validation=dict(validation)))
            transfers.append(dict(task=task, simulation_id=i, npz_sha256=sha(f"{task}-{i}"),
                                  json_sha256=sha(f"sidecar-{task}-{i}")))
    layout = {name: sha(name) for name in ("sample_ids", "sample_stems", "node_coords", "sample_node_coords", "frame_values")}
    for mech, task in (("UT", "UT"), ("FT", "FT"), ("MULTI", "UT"), ("MULTI", "FT")):
        banks.append(dict(mech_mode=mech, task=task, sample_ids=memberships[mech],
                          source_damage_sha256=[sha(f"{task}-{i}") for i in memberships[mech]],
                          samples=len(memberships[mech]), frames=21, struts=2319 if task == "UT" else 2259,
                          unavailable_sample_frames=0, samples_with_unavailable_trailing_frames=0,
                          observed_unknown_parent_frames=0, unknown_parent_frames=0,
                          npz_sha256=sha(mech+task), hpc_npz_sha256=sha(mech+task),
                          json_sha256=sha("json"+mech+task), hpc_json_sha256=sha("json"+mech+task),
                          field_layout_sha256=layout, local_field_layout_sha256=layout))
    return dict(schema_version="fcc_status_corpus_review_v1", manifest_sha256=sha("manifest"),
                code_sha256={name: sha(name) for name in ("batch_driver", "damage_core", "exporter")},
                attempts=rows, transfers=transfers, banks=banks, required_ids=memberships,
                excluded_ids={task: sorted(set(range(9001))-set(memberships[task])) for task in ("UT", "FT")})


class CorpusReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete = evidence()

    def setUp(self):
        self.evidence = copy.deepcopy(self.complete)
        self.execution = dict(dataset_manifest_sha256=self.evidence["manifest_sha256"], corpus_evidence=self.evidence)

    def summary(self, selected=()):
        return REVIEW["corpus_summary"](self.execution, selected)

    def test_bare_completion_flag_has_no_authority(self):
        self.assertIsNone(REVIEW["corpus_summary"]({"corpus_complete": True}))

    def test_complete_delivery_requires_all_evidence(self):
        result = self.summary()
        self.assertTrue(result["complete"])
        self.assertEqual(result["validated_products"], 18002)
        self.assertEqual(result["required_unavailable"], 0)

    def test_omitted_attempts_are_pending(self):
        self.evidence["attempts"] = self.evidence["attempts"][:1]
        self.evidence["transfers"] = self.evidence["transfers"][:1]
        self.evidence["banks"] = []
        result = self.summary()
        self.assertFalse(result["complete"])
        self.assertEqual(sum(v["pending"] for v in result["outcomes"].values()), 18001)

    def test_short_required_raw_source_keeps_its_actual_frame_count(self):
        validation = self.evidence["attempts"][601]["validation"]
        validation.update(frames=20, known_parent_frames=20*2319)
        for bank in self.evidence["banks"]:
            if bank["task"] == "UT":
                bank.update(unavailable_sample_frames=1, samples_with_unavailable_trailing_frames=1,
                            unknown_parent_frames=2319)
        result = self.summary()
        self.assertTrue(result["complete"])
        self.assertEqual(result["raw_frame_counts"]["UT"], {"20": 1, "21": 9000})
        self.assertEqual(result["short_raw_source_ids"], {"UT": [601], "FT": []})
        self.assertEqual(result["short_required_sources"], 1)
        self.assertEqual(result["raw_status_positions"], 21*18002-1)
        self.assertEqual([bank["frames"] for bank in result["banks"]], [21]*4)

    def test_bank_padding_and_observed_unknowns_must_match_raw_sources(self):
        validation = self.evidence["attempts"][601]["validation"]
        validation.update(frames=20, known_parent_frames=20*2319-1, missing_status=1)
        for bank in self.evidence["banks"]:
            if bank["task"] == "UT":
                bank.update(unavailable_sample_frames=1, samples_with_unavailable_trailing_frames=1,
                            observed_unknown_parent_frames=1, unknown_parent_frames=2320)
        result = self.summary()
        self.assertTrue(result["complete"])
        self.assertEqual(result["banks"][0]["observed_unknown_parent_frames"], 1)
        self.assertEqual(result["banks"][0]["unknown_parent_frames"], 2320)
        for field in ("unavailable_sample_frames", "samples_with_unavailable_trailing_frames",
                      "observed_unknown_parent_frames", "unknown_parent_frames"):
            with self.subTest(field=field):
                self.evidence["banks"][0][field] += 1
                with self.assertRaisesRegex(ValueError, "Bank observation/padding"):
                    self.summary()
                self.evidence["banks"][0][field] -= 1

    def test_zero_frame_raw_source_is_invalid(self):
        validation = self.evidence["attempts"][601]["validation"]
        validation.update(frames=0, known_parent_frames=0)
        with self.assertRaisesRegex(ValueError, "Invalid corpus validation"):
            self.summary()

    def test_duplicate_attempt_rejected(self):
        self.evidence["attempts"].append(self.evidence["attempts"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.summary()

    def test_exclusion_cannot_hide_required_failure(self):
        row = self.evidence["attempts"][0]
        row.update(state="unreadable", error="synthetic ODB read failure")
        self.evidence["transfers"] = self.evidence["transfers"][1:]
        self.evidence["banks"] = []
        result = self.summary()
        self.assertFalse(result["complete"])
        self.assertEqual(result["required_unavailable"], 1)
        self.evidence["excluded_ids"]["UT"].append(0)
        with self.assertRaisesRegex(ValueError, "Excluded IDs"):
            self.summary()

    def test_terminal_preexisting_exclusion_remains_visible(self):
        row = self.evidence["attempts"][9000]
        row.update(state="unreadable", error="synthetic excluded ODB failure")
        self.evidence["transfers"] = [r for r in self.evidence["transfers"] if (r["task"], r["simulation_id"]) != ("UT", 9000)]
        result = self.summary()
        self.assertTrue(result["complete"])
        self.assertEqual(result["outcomes"]["UT"]["unsuccessful"], 1)

    def test_wrong_receipt_or_bank_source_rejected(self):
        self.evidence["transfers"][0]["npz_sha256"] = sha("wrong")
        with self.assertRaisesRegex(ValueError, "Transfer"):
            self.summary()
        self.evidence["transfers"][0]["npz_sha256"] = sha("UT-0")
        self.evidence["banks"][0]["source_damage_sha256"][0] = sha("wrong")
        with self.assertRaisesRegex(ValueError, "Bank source"):
            self.summary()

    def test_bank_order_and_archive_bytes_must_match(self):
        bank = self.evidence["banks"][0]
        bank["sample_ids"] = bank["sample_ids"][::-1]
        with self.assertRaisesRegex(ValueError, "Bank order"):
            self.summary()
        bank["sample_ids"] = bank["sample_ids"][::-1]
        bank["hpc_npz_sha256"] = sha("wrong")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.summary()

    def test_skipped_preflight_requires_matching_selected_checksum(self):
        row = self.evidence["attempts"][0]
        prior = dict(row.pop("validation"), task="UT", simulation_id=0, source_file=row["output"],
                     npz_sha256=row["npz_sha256"], metadata={"batch": {"code_sha256": self.evidence["code_sha256"]}})
        row["state"] = "skipped_valid"
        self.assertTrue(self.summary([prior])["complete"])
        prior["npz_sha256"] = sha("wrong")
        with self.assertRaisesRegex(ValueError, "Skipped product"):
            self.summary([prior])

    def test_partial_csv_keeps_all_expected_rows(self):
        self.evidence["attempts"] = self.evidence["attempts"][:1]
        self.evidence["transfers"], self.evidence["banks"] = [], []
        with tempfile.TemporaryDirectory(prefix="nf-damage-report-test-") as temporary:
            root = Path(temporary)
            REVIEW["write_corpus"](self.summary(), root)
            self.assertEqual(len((root/"corpus-attempts.csv").read_text().splitlines()), 18003)
            self.assertTrue((root/"corpus-coverage.png").is_file())


if __name__ == "__main__":
    unittest.main()

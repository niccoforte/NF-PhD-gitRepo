"""Output-only Explicit replay preparation; no solver execution."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from validation_controls import (output_filter_replay_contents, prepare_output_filter_replays,
                                 output_filter_history_data, output_filter_replay_delivery,
                                 prepare_output_filter_precision_replay)
from validation_diagnostics import blocks, key


MODEL = """*Heading
** Synthetic source
*Node
1,0,0
2,0,1
*Element,type=B21
1,1,2
*Nset,nset=load
2,
*Nset,nset=Set-top
2,
*Material,name=ti
*Elastic
135165.,.428571
*Plastic
900.,0.
*Damage Initiation,criterion=DUCTILE
.1105,0.,0.
*Amplitude,name=Amp-1
0.,0.,.1,52.8
*Step,name=Step-1,nlgeom=YES
*Dynamic,Explicit,scale factor=.95
,.1
*Bulk Viscosity
.06,1.2
*Boundary,amplitude=Amp-1
{grip},1,1
{grip},2,2,1.
*Restart,write,number interval=1,time marks=NO
*Output,field
*Node Output
RF,UT
*Element Output
LE,STATUS
*Output,history
*Energy Output
ALLIE,ALLKE,ALLSE
*Node Output,nset={grip}
U2,RF2
*End Step
"""


class OutputFilterReplayTests(unittest.TestCase):
    def history_fixture(self, nodes=2):
        time = np.arange(11)*.0005
        regions = {}
        filters = {name: {} for name in ("PHYSICAL2FN", "SELECTED", "ENVELOPE100", "ENVELOPE50")}
        for label in range(1, nodes+1):
            channels = {}
            for suffix in [""]+["_"+name for name in filters]:
                for variable, scale in (("U2", 2.), ("RF2", float(label))):
                    channels[variable+suffix] = SimpleNamespace(data=list(zip(time, scale*time)))
            regions["Node PART-1-1."+str(label)] = SimpleNamespace(historyOutputs=channels)
        odb = SimpleNamespace(rootAssembly=SimpleNamespace(nodeSets={"SET-TOP": SimpleNamespace(
            nodes=(tuple(SimpleNamespace(label=i) for i in range(1, nodes+1)),))}),
            steps={"Step-1": SimpleNamespace(historyRegions=regions)})
        meta = dict(task="UT", grip_nset="Set-top", filters=filters,
                    raw_history_interval_s=.0005, filtered_history_interval_s=.0005)
        return odb, meta

    def test_every_grip_node_has_all_channels_and_pairs_aggregate_consistently(self):
        odb, meta = self.history_fixture()
        data, ledger = output_filter_history_data(odb, meta)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(set(data), {"RAW", *meta["filters"]})
        np.testing.assert_allclose(data["RAW"][:, 1], 2*data["RAW"][:, 0])
        np.testing.assert_allclose(data["SELECTED"][:, 2], 3*data["SELECTED"][:, 0])
        del odb.steps["Step-1"].historyRegions["Node PART-1-1.2"].historyOutputs["RF2_ENVELOPE50"]
        with self.assertRaisesRegex(ValueError, "channel"):
            output_filter_history_data(odb, meta)

    def test_misaligned_filtered_pair_is_rejected(self):
        odb, meta = self.history_fixture(1)
        target = odb.steps["Step-1"].historyRegions["Node PART-1-1.1"].historyOutputs["U2_SELECTED"]
        target.data[3] = (.0016, target.data[3][1])
        with self.assertRaisesRegex(ValueError, "aligned"):
            output_filter_history_data(odb, meta)

    def test_original_source_keeps_its_own_coarser_observed_sampling(self):
        odb, meta = self.history_fixture(1)
        meta["raw_history_interval_s"] = .00001
        values, _ = output_filter_history_data(odb, meta, include_filters=False)
        self.assertEqual(len(values["RAW"]), 11)

    def test_delivery_keeps_failed_status_and_rejects_changed_source_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary);queue = out/"queue";run = queue/"runs/replay";run.mkdir(parents=True)
            source=out/"original.inp";source.write_bytes(b"source")
            original=out/"original.odb";original.write_bytes(b"original ODB")
            inp=out/"replay.inp";inp.write_bytes(b"derived")
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            meta=dict(job="replay",input=str(inp),input_sha256=digest(inp),source_inp=str(source),
                      source_sha256=digest(source),source_odb=str(original),source_odb_sha256=digest(original))
            for suffix in (".inp", ".odb", ".dat", ".sta"):
                (run/("replay"+suffix)).write_bytes(inp.read_bytes() if suffix==".inp" else suffix.encode())
            receipt=dict(job="replay",required_outputs_missing=[],files=[dict(file=p.name,sha256=digest(p))
                         for p in run.iterdir()])
            (run/"delivery.json").write_text(json.dumps(receipt))
            state=dict(job="replay",status="failed",solver_returncode=1)
            proof=output_filter_replay_delivery(meta,state,queue)
            self.assertEqual(proof["solver_status"], "failed")
            self.assertFalse(proof["prefix_qualified"])
            (run/"replay.odb").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "delivery"):
                output_filter_replay_delivery(meta,state,queue)
            (run/"replay.odb").write_bytes(b".odb")
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source"):
                output_filter_replay_delivery(meta,state,queue)

    def test_only_output_cards_change_and_original_step_amplitude_survive(self):
        text = MODEL.format(grip="load")
        result, metadata = output_filter_replay_contents(text, "FT", 318.16)
        output_kinds = {"*filter", "*output", "*node output", "*element output", "*energy output"}
        original = [b.splitlines()[0:2] for b in blocks(text) if key(b) not in output_kinds]
        prepared = [b.splitlines()[0:2] for b in blocks(result) if key(b) not in output_kinds]
        self.assertEqual(original, prepared)
        self.assertIn("*Dynamic,Explicit,scale factor=.95\n,.1", result)
        self.assertIn("*Amplitude,name=Amp-1\n0.,0.,.1,52.8", result)
        self.assertEqual(metadata["source_period_s"], .1)
        self.assertEqual(metadata["grip_nset"], "load")
        self.assertEqual(result.count("*Filter,"), 4)
        self.assertEqual(result.count("*Output, history, filter="), 4)
        self.assertIn("*Output, history, time interval=1e-05", result)
        self.assertIn("*Node Output, nset=load\nU2, RF2", result)
        self.assertNotIn("*Element Output, filter=", result)
        self.assertIn("LE, S, PEEQ, SDEG, STATUS", result)

    def test_uat_filters_are_solver_rate_even_above_saved_nyquist(self):
        result, metadata = output_filter_replay_contents(MODEL.format(grip="Set-top"), "UT", 1836.34)
        self.assertEqual(metadata["grip_nset"], "Set-top")
        self.assertAlmostEqual(metadata["filters"]["PHYSICAL2FN"]["cutoff_Hz"], 1836.34)
        self.assertAlmostEqual(metadata["filters"]["SELECTED"]["cutoff_Hz"], 1000/3)
        self.assertIn("1836.34, 2", result)
        self.assertIn("*Node Output, nset=Set-top\nU2, RF2", result)
        self.assertTrue(all(v["order"] == 2 and v["startup"] == "DC"
                            for v in metadata["filters"].values()))

    def test_unknown_or_already_filtered_sources_are_rejected(self):
        text = MODEL.format(grip="load")
        for changed in (text.replace("*Dynamic,Explicit", "*Static"),
                        text.replace("*Output,field", "*Output,field,filter=OLD"),
                        text.replace("*Amplitude", "*Include,input=unseen.inp\n*Amplitude")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                output_filter_replay_contents(changed, "FT", 318.16)

    def test_two_source_manifest_is_ordered_hash_bound_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "audit/curve-processing/paired-examples/native-index.json"
            index.parent.mkdir(parents=True)
            rows = []
            for task, grip in (("UT", "Set-top"), ("FT", "load")):
                source = root / (task + ".inp")
                source.write_text(MODEL.format(grip=grip), encoding="ascii")
                source.with_suffix(".odb").write_bytes(b"immutable synthetic ODB")
                rows.append(dict(topology="hex", design="full", id=336, task=task,
                                 source_inp=str(source), source_odb=str(source.with_suffix(".odb")),
                                 source_inp_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            index.write_text(json.dumps(rows))
            out = root / "audit/curve-processing/output-filter-replays"
            result = prepare_output_filter_replays(root, out)
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual([j["job"] for j in manifest["jobs"]], ["of_hex336_ft", "of_hex336_ut"])
            self.assertTrue(all(j["cpus"] == 8 and j["allow_solver_failure"] for j in manifest["jobs"]))
            for job in manifest["jobs"]:
                self.assertEqual(hashlib.sha256(Path(job["input"]).read_bytes()).hexdigest(), job["input_sha256"])
            before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
            prepare_output_filter_replays(root, out)
            self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
            target = Path(manifest["jobs"][0]["input"])
            target.write_text("changed")
            with self.assertRaisesRegex(ValueError, "immutable"):
                prepare_output_filter_replays(root, out)

    def test_double_precision_attempt_copies_input_bytes_and_retains_parent_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);parent=base/"parent";parent.mkdir()
            source=base/"original.inp";source.write_bytes(b"original")
            odb=base/"original.odb";odb.write_bytes(b"source ODB")
            inp=parent/"single.inp";inp.write_bytes(b"*Heading\r\n** unchanged exact bytes\r\n")
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            meta=dict(job="single",task="FT",input=str(inp),input_sha256=sha(inp),
                      source_inp=str(source),source_sha256=sha(source),source_odb=str(odb),source_odb_sha256=sha(odb))
            (parent/"source.json").write_text(json.dumps(meta))
            result=prepare_output_filter_precision_replay(parent,base/"double")
            manifest=json.loads(Path(result["manifest"]).read_text());job=manifest["jobs"][0]
            self.assertEqual(Path(job["input"]).read_bytes(),inp.read_bytes())
            self.assertEqual(job["execution_options"],["double=both","output_precision=full"])
            child=json.loads((Path(job["input"]).parent/"source.json").read_text())
            self.assertEqual(child["parent_job"],"single")
            self.assertEqual(child["parent_input_sha256"],sha(inp))
            prepare_output_filter_precision_replay(parent,base/"double")


if __name__ == "__main__":
    unittest.main()

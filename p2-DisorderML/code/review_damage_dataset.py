"""Review explicit FIELDd products; no ODB access, HPC calls or model evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from resources.damage import require, validate_status_arrays


def corpus_summary(execution, selected=()):
    """Reconcile supplied FCC evidence; never query archives or infer missing success.

    Optional execution.corpus_evidence schema fcc_status_corpus_review_v1:
    - checked_utc optionally identifies the evidence snapshot.
    - manifest_sha256 equals execution.dataset_manifest_sha256; code_sha256 maps
      batch_driver/damage_core/exporter to the pinned production hashes.
    - attempts: task, simulation_id, output, state; produced/skipped_valid also
      need npz_sha256 and the exporter's nine validation integers (frames,
      struts, elements, known_parent_frames, failed_parent_frames,
      final_failed_struts, final_failed_elements, missing_status,
      status_reactivations). skipped_valid may instead reuse a selected record
      with matching identity, filename, checksum and all three code pins.
      Raw frames must be positive; shorter histories remain explicit and are
      distinct from the processed banks' 21-frame axes.
      failed/unreadable/unsupported require error; running/pending are allowed.
      Omitted expected identities are pending, never inferred successful.
    - required_ids: ordered integer UT/FT/MULTI arrays from existing allFIELD
      membership; excluded_ids: exact UT/FT complements in 0..9000.
    - transfers: checksum-verified task, simulation_id, npz_sha256, json_sha256.
    - banks: mech_mode, task, samples, frames, struts, ordered sample_ids and
      source_damage_sha256; local npz_sha256/json_sha256 must equal corresponding
      hpc_npz_sha256/hpc_json_sha256. field_layout_sha256 is the bank metadata's
      original allFIELD member hashes; local_field_layout_sha256 comes from
      read_field_layout(...)["provenance"]["member_sha256"]. These dictionaries
      contain sample_ids/sample_stems/node_coords/sample_node_coords/frame_values
      and must match exactly. Source hashes must join task/ID transfer receipts.
      Nonnegative unavailable_sample_frames, samples_with_unavailable_trailing_frames,
      observed_unknown_parent_frames and unknown_parent_frames must reconcile
      the raw source prefixes and the bank's masked trailing frames.

    The evidence producer reconciles manifest/code pins and actual files before
    supplying rows. No filesystem proof is invented by this plotting consumer.
    A standalone corpus_complete flag has no authority here.
    """
    evidence = execution.get("corpus_evidence")
    if evidence is None:
        return None
    require(evidence["schema_version"] == "fcc_status_corpus_review_v1", "Unsupported corpus evidence")
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
    require(digest(evidence["manifest_sha256"]) and
            evidence["manifest_sha256"] == execution.get("dataset_manifest_sha256"), "Corpus manifest pin mismatch")
    code = evidence["code_sha256"]
    require(set(code) == {"batch_driver", "damage_core", "exporter"} and
            all(digest(value) for value in code.values()), "Invalid corpus code pins")
    expected = {(task, i) for task in ("UT", "FT") for i in range(9001)}
    memberships = evidence["required_ids"]
    for task, count in (("UT", 8882), ("FT", 8928), ("MULTI", 8811)):
        values = memberships[task]
        require(len(values) == len(set(values)) == count and 0 in values and
                all(type(i) is int and 0 <= i <= 9000 for i in values), "Invalid ordered "+task+" membership")
    require(set(memberships["MULTI"]) <= set(memberships["UT"]) & set(memberships["FT"]), "MULTI membership differs")
    required = {(task, i) for task in ("UT", "FT") for i in memberships[task]}
    for task in ("UT", "FT"):
        excluded = evidence["excluded_ids"][task]
        require(len(excluded) == len(set(excluded)) and set(excluded) ==
                set(range(9001))-set(memberships[task]), "Excluded IDs differ from existing "+task+" membership")
    selected = {(r["task"], r["simulation_id"]): r for r in selected}
    attempts, outputs = {}, set()
    successful, unsuccessful = {"produced", "skipped_valid"}, {"failed", "unreadable", "unsupported"}
    fields = ("frames", "struts", "elements", "known_parent_frames", "failed_parent_frames",
              "final_failed_struts", "final_failed_elements", "missing_status", "status_reactivations")
    for supplied in evidence["attempts"]:
        row = dict(supplied)
        key = row["task"], row["simulation_id"]
        require(type(key[1]) is int and key in expected and key not in attempts, "Duplicate or unexpected attempt identity")
        require(isinstance(row["output"], str) and row["output"] and row["output"] not in outputs,
                "Missing or duplicate attempt output")
        require(row["state"] in successful | unsuccessful | {"pending", "running"}, "Unsupported attempt state")
        outputs.add(row["output"])
        if row["state"] in successful:
            require(digest(row.get("npz_sha256")), "Validated attempt needs a product checksum")
            validation = row.get("validation")
            if validation is None and row["state"] == "skipped_valid":
                prior = selected.get(key, {})
                require(prior.get("npz_sha256") == row["npz_sha256"] and
                        prior.get("metadata", {}).get("batch", {}).get("code_sha256") == code and
                        Path(prior.get("source_file", "")).name == Path(row["output"]).name,
                        "Skipped product lacks matching selected validation")
                validation = {name: prior[name] for name in fields}
            require(isinstance(validation, dict) and all(type(validation.get(k)) is int and validation[k] >= 0
                                                       for k in fields), "Incomplete attempt validation")
            v = validation
            require(v["frames"] > 0 and v["struts"] == {"UT": 2319, "FT": 2259}[key[0]] and v["elements"] > 0 and
                    v["failed_parent_frames"] <= v["known_parent_frames"] <= v["frames"]*v["struts"] and
                    v["final_failed_struts"] <= min(v["struts"], v["failed_parent_frames"]) and
                    v["final_failed_elements"] <= v["elements"] and v["missing_status"] <= v["frames"]*v["elements"] and
                    v["status_reactivations"] == 0, "Invalid corpus validation counts")
            row["validation"] = validation
        elif row["state"] in unsuccessful:
            require(isinstance(row.get("error"), str) and row["error"].strip(), "Unsuccessful attempt needs its reason")
        if row["state"] not in successful:
            row.pop("validation", None)
        attempts[key] = row
    valid = {key: row for key, row in attempts.items() if row["state"] in successful}
    transfers = {}
    for row in evidence.get("transfers", []):
        key = row["task"], row["simulation_id"]
        require(key in valid and key not in transfers and row["npz_sha256"] == valid[key]["npz_sha256"] and
                digest(row["json_sha256"]), "Transfer does not match a unique validated product")
        transfers[key] = row
    banks = {}
    layout_keys = {"sample_ids", "sample_stems", "node_coords", "sample_node_coords", "frame_values"}
    for row in evidence.get("banks", []):
        key = row["mech_mode"], row["task"]
        require(key in {("UT", "UT"), ("FT", "FT"), ("MULTI", "UT"), ("MULTI", "FT")} and key not in banks,
                "Duplicate or unexpected bank")
        require(row["sample_ids"] == memberships[key[0]], "Bank order differs from existing allFIELD IDs")
        require(row["samples"] == len(row["sample_ids"]) and row["frames"] == 21 and
                row["struts"] == {"UT": 2319, "FT": 2259}[key[1]], "Bank axes differ from the required population")
        for kind in ("npz", "json"):
            require(digest(row[kind+"_sha256"]) and row[kind+"_sha256"] == row["hpc_"+kind+"_sha256"],
                    "Local/HPC bank checksum mismatch")
        layout = row["field_layout_sha256"]
        require(set(layout) == layout_keys and all(digest(v) for v in layout.values()) and
                layout == row["local_field_layout_sha256"], "Bank/allFIELD layout hash mismatch")
        hashes = row["source_damage_sha256"]
        require(len(hashes) == len(row["sample_ids"]) and all((key[1], i) in transfers and
                value == transfers[(key[1], i)]["npz_sha256"] for i, value in zip(row["sample_ids"], hashes)),
                "Bank source hash does not match its FIELDd receipt")
        sources = [valid[(key[1], i)]["validation"] for i in row["sample_ids"]]
        require(all(v["frames"] <= 21 for v in sources), "Bank source exceeds the processed frame axis")
        coverage = dict(unavailable_sample_frames=sum(21-v["frames"] for v in sources),
                        samples_with_unavailable_trailing_frames=sum(v["frames"] < 21 for v in sources),
                        observed_unknown_parent_frames=sum(v["frames"]*v["struts"]-v["known_parent_frames"] for v in sources))
        coverage["unknown_parent_frames"] = coverage["observed_unknown_parent_frames"]+coverage["unavailable_sample_frames"]*row["struts"]
        require(all(type(row.get(name)) is int and row[name] >= 0 and row[name] == expected_count
                    for name, expected_count in coverage.items()), "Bank observation/padding counts differ from its sources")
        banks[key] = row
    rows = [dict(attempts.get(key, dict(task=key[0], simulation_id=key[1], output="", state="pending")),
                 required_for_MLdata=key in required) for key in sorted(expected)]
    counts = {task: dict(validated=sum(key[0] == task for key in valid),
                        unsuccessful=sum(r["task"] == task and r["state"] in unsuccessful for r in rows),
                        pending=sum(r["task"] == task and r["state"] in {"pending", "running"} for r in rows))
              for task in ("UT", "FT")}
    metrics = {name: sum(r["validation"][name] for r in valid.values())
               for name in ("known_parent_frames", "failed_parent_frames", "missing_status", "status_reactivations")}
    metrics.update(raw_status_positions=sum(r["validation"]["frames"]*r["validation"]["elements"] for r in valid.values()),
                   unknown_parent_frames=sum(r["validation"]["frames"]*r["validation"]["struts"]-
                                             r["validation"]["known_parent_frames"] for r in valid.values()))
    raw_frames = {task: [row["validation"]["frames"] for key, row in valid.items() if key[0] == task]
                  for task in ("UT", "FT")}
    short = {key for key, row in valid.items() if row["validation"]["frames"] < 21}
    complete = (len(attempts) == len(expected) and not any(c["pending"] for c in counts.values()) and
                required <= valid.keys() and valid.keys() == transfers.keys() and len(banks) == 4)
    return dict(complete=complete, checked_utc=evidence.get("checked_utc"), attempts=rows, outcomes=counts, **metrics,
                expected_attempts=len(expected), explicit_attempts=len(attempts),
                validated_products=len(valid), transfer_verified_products=len(transfers), verified_banks=len(banks),
                untransferred_products=len(valid)-len(transfers),
                required_unavailable=sum(key not in valid for key in required),
                raw_frame_counts={task: {str(n): values.count(n) for n in sorted(set(values))}
                                  for task, values in raw_frames.items()},
                short_raw_source_ids={task: sorted(i for t, i in short if t == task) for task in ("UT", "FT")},
                short_raw_sources=len(short), short_required_sources=len(short & required),
                excluded_ids=evidence["excluded_ids"], manifest_sha256=evidence["manifest_sha256"], code_sha256=code,
                banks=[{name: row[name] for name in ("mech_mode", "task", "samples", "frames", "struts", "npz_sha256", "json_sha256",
                                                   "unavailable_sample_frames", "samples_with_unavailable_trailing_frames",
                                                   "observed_unknown_parent_frames", "unknown_parent_frames")}
                       for row in banks.values()])


def write_corpus(summary, output_dir):
    """Write explicit pending/exception rows and a corpus-only quantitative figure."""
    fields = ["task", "simulation_id", "output", "state", "required_for_MLdata", "npz_sha256", "error",
              "frames", "struts", "elements", "known_parent_frames", "failed_parent_frames", "final_failed_struts",
              "final_failed_elements", "missing_status", "status_reactivations"]
    with (output_dir/"corpus-attempts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(row, **row.get("validation", {})) for row in summary["attempts"])
    (output_dir/"corpus-summary.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "attempts"}, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bottom = np.zeros(2)
    for state, color in (("validated", "#3b8b6d"), ("unsuccessful", "#d62728"), ("pending", "#999999")):
        values = [summary["outcomes"][task][state] for task in ("UT", "FT")]
        axes[0].bar(["UT", "FT"], values, bottom=bottom, label=state, color=color)
        bottom += values
    axes[0].set(title="Processing coverage · 9,001 expected/task", ylabel="Specimens")
    axes[0].legend(fontsize=8)
    fractions = {task: [100*r["validation"]["final_failed_struts"]/r["validation"]["struts"]
                       for r in summary["attempts"]
                       if r["task"] == task and r["state"] in {"produced", "skipped_valid"}]
                 for task in ("UT", "FT")}
    bins = np.histogram_bin_edges([v for values in fractions.values() for v in values], bins=25)
    for task, color in (("UT", "#3b528b"), ("FT", "#c25419")):
        if fractions[task]:
            axes[1].hist(fractions[task], bins=bins, histtype="step", color=color,
                         label=f"{task} · n={len(fractions[task]):,}")
    axes[1].set(title="Successfully validated raw specimens only", xlabel="Observed final fractured struts (%)", ylabel="Specimens")
    if summary["validated_products"]:
        axes[1].legend(fontsize=8)
    fig.suptitle("FCC corpus evidence · "+("reconciled" if summary["complete"] else "partial; delivery incomplete"))
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output_dir/"corpus-coverage.png", dpi=170)
    plt.close(fig)
    lines = [f"**Corpus evidence {'reconciled' if summary['complete'] else 'partial'}:** "
            f"{summary['validated_products']:,}/18,002 products validated; "
            f"{summary['transfer_verified_products']:,} checksum-verified local products; "
            f"{summary['verified_banks']}/4 verified banks. "
            f"{summary['required_unavailable']:,} required source products remain unvalidated.", "",
            f"Recorded batch validation covers {summary['raw_status_positions']:,} element/frame STATUS positions, "
            f"{summary['missing_status']:,} missing observations and {summary['unknown_parent_frames']:,} unknown parent/frame states. "
            "Terminal exceptions remain in the attempt table even when already excluded from MLdata.", "",
            "Raw saved-frame counts (frames: specimens): "+"; ".join(
                task+" "+(", ".join(f"{frames}: {count:,}" for frames, count in summary["raw_frame_counts"][task].items()) or "none")
                for task in ("UT", "FT"))+". "
            f"{summary['short_raw_sources']:,} raw histories have fewer than 21 frames "
            f"({summary['short_required_sources']:,} required sources). "
            "Their IDs and actual frame counts remain explicit in the supporting table/JSON; "
            "observed final fracture means each source's last available frame. "
            "A short raw history is distinct from the processed 21-frame bank axis and is never extended by this report.", "",
            "![Corpus processing coverage and fracture distribution](corpus-coverage.png)", "",
            "The distribution divides failed struts by each task's native count (UT2,319; FT2,259). "
            "The separate native ODB comparison remains bounded to eight preflight products; "
            "these corpus totals use recorded batch validation.", "",
            "[All expected attempts and exceptions](corpus-attempts.csv) · [Corpus totals and full bank hashes](corpus-summary.json)", ""]
    if summary["checked_utc"]:
        lines = [f"Evidence snapshot: {summary['checked_utc']}", ""]+lines
    if not summary["complete"]:
        lines += ["Pending includes expected identities whose results have not yet been reconciled here. "
                  "Running-job counts and additional preflight products are recorded separately in the execution record.", ""]
    if summary["banks"]:
        lines += ["Verified bank delivery preserves existing ordered memberships and periodic0. "
                  "Unknown strut/frame states include both unknown observed states and absent trailing sample frames; "
                  "their separate counts are retained in the compact JSON.", "",
                  "| Bank | Samples | Frames | Native struts | Absent sample frames | Unknown strut/frame states | Local/HPC NPZ SHA256 (prefix) |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        lines += [f"| {r['mech_mode']}-{r['task']} | {r['samples']:,} | {r['frames']} | {r['struts']:,} | "
                  f"{r['unavailable_sample_frames']:,} | {r['unknown_parent_frames']:,} | {r['npz_sha256'][:12]} |"
                  for r in summary["banks"]]
        lines += [""]
    return lines+["The geometry and frame-evolution plots below show selected examples only.", ""]


def review(products, output_dir, execution_json=None):
    """Build a conclusions-first visual report from validated selected products."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    execution = json.loads(Path(execution_json).read_text()) if execution_json else {}
    require(isinstance(execution, dict), "Execution record must be a JSON object")
    cases, identities = [], set()
    for path in map(Path, products):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        summary = validate_status_arrays(arrays)
        meta = json.loads(str(arrays["metadata_json"]))
        identity = str(arrays["task"]), int(arrays["simulation_id"])
        require(identity not in identities, "Duplicate review task/sample identity")
        identities.add(identity)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b""):
                digest.update(block)
        record = dict(task=identity[0], simulation_id=identity[1], **summary,
                      final_unknown_struts=int((~arrays["strut_failure_known"][-1]).sum()),
                      frame_time_start_s=float(arrays["frame_time"][0]),
                      frame_time_end_s=float(arrays["frame_time"][-1]),
                      maximum_FIELDu_time_offset_s=meta.get("batch", {}).get("field_alignment", {}).get("maximum_time_difference_s"),
                      source_file=str(path.resolve()), npz_sha256=digest.hexdigest(), metadata=meta)
        cases.append((arrays, record))
    require(cases, "At least one explicit FIELDd product is required")
    cases.sort(key=lambda case: (case[1]["task"] != "UT", case[1]["simulation_id"]))
    ids = sorted({record["simulation_id"] for _, record in cases})
    colors = dict(active="#440154", failed="#d62728", unknown="#111111")
    fig, axes = plt.subplots(2, len(ids), figsize=(max(5, 3.4*len(ids)), 9.8), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for arrays, record in cases:
        ax = axes[0 if record["task"] == "UT" else 1, ids.index(record["simulation_id"])]
        ax.set_visible(True)
        known, failed = arrays["strut_failure_known"][-1], arrays["strut_failed"][-1]
        for state, mask, width, style in (("active", known & ~failed, .45, "solid"),
                                          ("failed", known & failed, 1.7, "solid"),
                                          ("unknown", ~known, 1.1, "dashed")):
            ax.add_collection(LineCollection(arrays["endpoint_xy0"][mask], colors=colors[state],
                                              linewidths=width, linestyles=style, zorder=3 if state != "active" else 1))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"{record['task']} · {'periodic 0' if record['simulation_id'] == 0 else 'ID '+str(record['simulation_id'])}\n"
                     f"{record['final_failed_struts']}/{record['struts']} fractured · {record['final_unknown_struts']} unknown", fontsize=10)
        ax.set_xlabel("Initial x (mm)")
        ax.set_ylabel("Initial y (mm)")
        ax.tick_params(labelsize=8)
    legend = [Line2D([0], [0], color=colors["active"], lw=2, label="D=0: all members known active"),
              Line2D([0], [0], color=colors["failed"], lw=2, label="D=1: at least one failed member"),
              Line2D([0], [0], color=colors["unknown"], lw=2, ls="--", label="Unknown: incomplete STATUS, no observed failure")]
    fig.suptitle("FCC fracture at the final saved frame\nWhole-element STATUS only", fontsize=13)
    fig.legend(handles=legend, loc="lower center", ncol=1, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .105, 1, .95), h_pad=3.2, w_pad=1.3)
    fig.savefig(output_dir/"fracture-final.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), squeeze=False, sharex="col")
    for column, task in enumerate(("UT", "FT")):
        for arrays, record in cases:
            if record["task"] != task:
                continue
            frame = arrays["frame_index"]
            label = "periodic 0" if record["simulation_id"] == 0 else "ID "+str(record["simulation_id"])
            axes[0, column].plot(frame, arrays["strut_failed"].sum(axis=1), marker=".", label=label)
            axes[1, column].plot(frame, 100*arrays["strut_failure_known"].mean(axis=1), marker=".", label=label)
        axes[0, column].set_title(task+" · observed fractured struts")
        axes[0, column].set_ylabel("Struts")
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_ylabel("Known fracture states (%)")
        axes[1, column].set_xlabel("Saved frame index (initial frame = 0)")
        axes[1, column].set_ylim(-2, 102)
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    fig.suptitle("Saved-frame evolution and observation coverage · no temporal interpolation")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_dir/"fracture-frames.png", dpi=170)
    plt.close(fig)
    records = [record for _, record in cases]
    keys = [key for key in records[0] if key != "metadata"]
    with (output_dir/"product-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: record[key] for key in keys} for record in records)
    corpus = corpus_summary(execution, records)
    (output_dir/"results.json").write_text(json.dumps(dict(products=records, execution=execution), indent=2)+"\n", encoding="utf-8")
    complete = corpus is not None and corpus["complete"]
    scope = ("Whole-corpus processing and archive delivery are reconciled by the supplied evidence. The figures below distinguish corpus totals from selected examples."
             if complete else "Corpus evidence is partial; processing or archive delivery remains incomplete. Geometry plots show selected products only."
             if corpus is not None else "This is a selected-product/preflight report. Whole-corpus completion has not been established by this report.")
    missing = sum(record["missing_status"] for record in records)
    unknown = sum(record["frames"]*record["struts"]-record["known_parent_frames"] for record in records)
    lines = ["# FCC STATUS dataset processing", "", scope, ""]
    if corpus is not None:
        lines += write_corpus(corpus, output_dir)
    lines += [
             f"**{len(records)} products validated:** {missing:,} missing element/frame STATUS records; "
             f"{unknown:,} unknown parent/frame fracture states. No observed reactivation is accepted.", "",
             "Fracture is binary: **D=1** when any constituent has whole-element STATUS=0; **D=0** when every constituent "
             "is known active. Missing observations remain unknown. SDEG is not used. UT excludes grips; initial FT gaps are absent geometry, not failed struts.", "",
             "![Final fracture states](fracture-final.png)", "",
             "| Task | ID | Frames | Struts | Body elements | Final fractured | Final unknown | Missing STATUS | Final time (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['task']} | {r['simulation_id']} | {r['frames']} | {r['struts']} | {r['elements']} | "
                     f"{r['final_failed_struts']} | {r['final_unknown_struts']} | {r['missing_status']} | {r['frame_time_end_s']:.7g} |")
    lines += ["", "![Frame evolution and known coverage](fracture-frames.png)", "",
              "Each product retains the initial frame and its recorded times. Dataset alignment follows the existing displacement-field "
              "saved-frame index/order; small specimen-time differences do not trigger interpolation. Actual times and any measured FIELDu offsets remain in the supporting records.", "",
              "Processing validation establishes exported labels and identity checks. It does not establish ML accuracy, training performance, "
              "fracture between recorded frames, or the statistical representativeness of these selected examples.", "",
              "[Product table](product-summary.csv) · [Source fingerprints, metadata and execution record](results.json)"]
    if execution:
        lines += ["", "<details>", "<summary>Execution records and periodic-source identity</summary>", "",
                  "The supplied record is reproduced below; this report does not query job states.",
                  "", "```json", json.dumps({k: v for k, v in execution.items() if k != "corpus_evidence"}, indent=2),
                  "```", "", "Full supplied corpus evidence, when present, is retained in results.json.", "", "</details>"]
    (output_dir/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", required=True, help="Explicit strut_status_v1 FIELDd NPZ; repeat")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1]/"samples"/"damage-dataset"))
    parser.add_argument("--execution-json", help="Optional supplied job/status and periodic-identity record")
    args = parser.parse_args()
    records = review(args.product, args.output_dir, args.execution_json)
    print(json.dumps(dict(report=str((Path(args.output_dir)/"README.md").resolve()), products=len(records))))


if __name__ == "__main__":
    main()

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
    (output_dir/"results.json").write_text(json.dumps(dict(products=records, execution=execution), indent=2)+"\n", encoding="utf-8")
    complete = execution.get("corpus_complete") is True
    scope = ("The execution record reports whole-corpus processing complete. The figures below show selected products only."
             if complete else "This is a selected-product/preflight report. Whole-corpus completion has not been established by this report.")
    missing = sum(record["missing_status"] for record in records)
    unknown = sum(record["frames"]*record["struts"]-record["known_parent_frames"] for record in records)
    lines = ["# FCC STATUS dataset processing", "", scope, "",
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
                  "", "```json", json.dumps(execution, indent=2), "```", "", "</details>"]
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

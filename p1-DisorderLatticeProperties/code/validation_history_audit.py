"""Read-only native Ti validation history audit (run with Abaqus Python).

This deliberately does not use the fixed-length production exporter: native
timestamps, absent observations and recorded STATUS=0 remain distinguishable.
No ODB upgrade, FEA submission or source-data write is performed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


DEFAULT_CASES = (
    "10/0.2/Fracture-hex-60-per-1.odb",
    "10/0.2/Fracture-tri-70-per-1.odb",
    "10/0.2/Fracture-kagome-40-per-1.odb",
    "3/0.16/Fracture-hex-80-per-1.odb",
    "10/0.2/Fracture-kagome-40-per-1.odb",
)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    stat = os.stat(path)
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()}


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def json_dump(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(item):
            return None
        return item
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(safe(value), handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def numeric_summary(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"n": len(values), "nonfinite": int(len(values) - len(finite)),
            "min": float(np.min(finite)) if len(finite) else None,
            "max": float(np.max(finite)) if len(finite) else None,
            "nonzero": int(np.count_nonzero(finite))}


def field_summary(field):
    """Use blocks for efficient scalar/finite scans; retain location separately."""
    arrays = [np.asarray(block.data, dtype=float).reshape(-1)
              for block in field.bulkDataBlocks]
    return numeric_summary(np.concatenate(arrays) if arrays else [])


def first_positive_locations(field, limit=12):
    locations = []
    for value in field.values:
        data = np.asarray(value.data, dtype=float).reshape(-1)
        if np.any(np.isfinite(data) & (data > 0)):
            locations.append({"instance": value.instance.name,
                              "element": getattr(value, "elementLabel", None),
                              "section_point": getattr(value.sectionPoint, "number", None),
                              "value": data.tolist()})
            if len(locations) >= limit:
                break
    return locations


def energy_window(rows, end, inclusive, name):
    """Raw IE>0 window, plus a labelled denominator/startup sensitivity window."""
    if end is None:
        return None
    eligible = [row for row in rows if (row[0] <= end if inclusive else row[0] < end)
                and row[2] > 0 and all(math.isfinite(x) for x in row)]
    if not eligible:
        return {"name": name, "samples": 0}
    max_ie = max(row[2] for row in eligible)
    stable = [row for row in eligible if row[0] >= .1 * end and row[2] > .01 * max_ie]
    def stats(selected):
        ratios = np.array([row[3] for row in selected])
        high = [row for row in selected if row[3] > .05]
        if not len(selected):
            return {"samples": 0}
        worst = max(selected, key=lambda row: row[3])
        return {"samples": len(selected), "ke_ie_max": float(np.max(ratios)),
                "ke_ie_q50_q90_q95_q99": np.quantile(ratios, [.5, .9, .95, .99]).tolist(),
                "over_5pct_count": len(high), "first_over_5pct": list(high[0]) if high else None,
                "worst": list(worst), "last": list(selected[-1])}
    return {"name": name, "end_time": end, "includes_event": inclusive,
            "positive_IE": stats(eligible),
            "startup_sensitivity_t_ge_0p1_event_IE_gt_1pct_max": stats(stable)}


def summarize_outputs(out):
    """Derive explicit event brackets/window diagnostics; plot in standard Python."""
    records = []
    plot_records = []
    for path in sorted(out.rglob("summary.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        curve_path = path.parent / "native_curve.csv"
        if curve_path.exists():
            with open(curve_path, newline="", encoding="utf-8") as handle:
                curve_reader = csv.reader(handle)
                curve_header = next(curve_reader)
                curve_rows = [[float(v) for v in row] for row in curve_reader]
            if "unknown_STATUS" not in curve_header:
                curve_header.append("unknown_STATUS")
                curve_rows = [row + [rec["status_region_count"] - row[4]] for row in curve_rows]
                write_csv(curve_path, curve_header, curve_rows)
            finite_load = [math.isfinite(row[1]) and math.isfinite(row[2]) for row in curve_rows]
            rec["nonfinite_load_history_indices"] = [i for i, known in enumerate(finite_load) if not known]
            rec["first_monitored_deletion_index"] = next((i for i, row in enumerate(curve_rows) if row[3] > 0), None)
            event_index = rec["first_monitored_deletion_index"]
            rec["last_intact_history_index"] = event_index - 1 if event_index is not None and event_index > 0 else None
            np.savez_compressed(str(path.parent / "native_curve.npz"),
                                time=np.array([r[0] for r in curve_rows]), U2=np.array([r[1] for r in curve_rows]),
                                RF2=np.array([r[2] for r in curve_rows]), finite_load_mask=np.array(finite_load))
        frame_path = path.parent / "field_frames.json"
        events = {}
        if frame_path.exists():
            frames = json.loads(frame_path.read_text(encoding="utf-8"))
            for name, variable, predicate in (
                    ("first_section_SDEG_positive", "SDEG", lambda row: row["max"] > 0),
                    ("first_DUCTCRT_ge_one", "DUCTCRT", lambda row: row["max"] >= 1),
                    ("first_global_field_STATUS_zero", "STATUS", lambda row: row["min"] == 0)):
                found = [i for i, row in enumerate(frames) if variable in row
                         and row[variable]["max"] is not None and predicate(row[variable])]
                if found:
                    index = found[0]
                    events[name] = {"first_frame_time": frames[index]["time"],
                                    "previous_frame_time": frames[index - 1]["time"] if index else None,
                                    "summary": frames[index][variable]}
            rec["field_event_brackets"] = events
        energy_path = path.parent / "native_energy.csv"
        deletion = rec["first_monitored_deletion_time"]
        if energy_path.exists():
            with open(energy_path, newline="", encoding="utf-8") as handle:
                rows = [[float(v) for v in row] for row in list(csv.reader(handle))[1:]]
            rows = [(r[0], r[1], r[2], r[1] / r[2] if r[2] else float("nan")) for r in rows]
            write_csv(energy_path, ["time_s", "ALLKE_Nmm", "ALLIE_Nmm", "KE_over_IE"], rows)
            rec["explicit_predeletion_energy_windows"] = []
            for name, start in (("final_20pct", .8 * deletion if deletion else None),
                                ("final_50pct", .5 * deletion if deletion else None),
                                ("t_ge_0p02_s", .02), ("t_ge_0p03_s", .03)):
                if deletion is None or start is None:
                    continue
                selected = [row for row in rows if start <= row[0] < deletion and row[2] > 0
                            and all(math.isfinite(v) for v in row)]
                ratios = [row[3] for row in selected]
                rec["explicit_predeletion_energy_windows"].append({
                    "name": name, "start_s": start, "end_exclusive_s": deletion, "samples": len(selected),
                    "max_ke_ie": max(ratios) if ratios else None,
                    "over_5pct_count": sum(r > .05 for r in ratios),
                    "last": selected[-1] if selected else None})
            if events and any(Path(rec["source"]).as_posix().endswith(relative)
                              for relative in DEFAULT_CASES):
                plot_records.append((rec, rows))
        json_dump(path, rec)
        records.append(rec)
    json_dump(out / "native_history_audit.json", records)
    if plot_records:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, (rec, rows) in zip(axes.flat, plot_records):
            ax.plot([r[0] for r in rows if r[2] > 0], [100 * r[3] for r in rows if r[2] > 0], lw=1.15, color="0.2", label="Native KE/IE")
            ax.axhline(5, color="tab:orange", ls="--", label="5%")
            if rec["first_monitored_deletion_time"] is not None:
                ax.axvline(rec["first_monitored_deletion_time"], color="tab:red", ls="-.", label="First monitored deletion")
            event = rec["field_event_brackets"].get("first_section_SDEG_positive")
            if event:
                ax.axvspan(event["previous_frame_time"] or 0, event["first_frame_time"], color="tab:blue", alpha=.3, label="First section SDEG>0 bracket")
            ax.set_yscale("log")
            source = Path(rec["source"])
            ax.set_title(source.stem + "  l=" + source.parent.parent.name + " mm, rho=" + source.parent.name)
            ax.set_xlabel("Native step time (s)")
            ax.set_ylabel("ALLKE / ALLIE (%)")
            ax.grid(True, which="both", alpha=.2)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle("Native energies: full startup shown; element deletion and section degradation are separate events")
        fig.savefig(out / "native_energy_detailed.png", dpi=180)
        plt.close(fig)
    return records


def refresh_global_event_locations(out):
    """Read first global-deletion field labels for the detailed cases only."""
    from odbAccess import openOdb

    for summary_path in sorted(out.rglob("summary.json")):
        rec = json.loads(summary_path.read_text(encoding="utf-8"))
        if not rec.get("last_two_field_summaries"):
            continue
        frames = json.loads((summary_path.parent / "field_frames.json").read_text(encoding="utf-8"))
        found = [row for row in frames if row.get("STATUS", {}).get("min") == 0]
        if not found:
            continue
        path = Path(rec["source"])
        before = fingerprint(path)
        odb = openOdb(path=str(path), readOnly=True)
        try:
            frame = odb.steps["Step-1"].frames[found[0]["index"]]
            known = {key[len("Element "):] for key in np.load(str(summary_path.parent / "native_status.npz"))["regions"].tolist()}
            locations = []
            for value in frame.fieldOutputs["STATUS"].values:
                if np.any(np.asarray(value.data) == 0):
                    key = value.instance.name + "." + str(value.elementLabel)
                    locations.append({"instance": value.instance.name, "element": value.elementLabel,
                                      "monitored_history": key in known})
            rec["global_STATUS_first_zero_check"] = {
                "first_field_index": found[0]["index"], "first_field_time": frame.frameValue,
                "model_elements_by_instance": {name: len(instance.elements) for name, instance in odb.rootAssembly.instances.items()},
                "observations": locations,
                "unique_elements": len(set((v["instance"], v["element"]) for v in locations)),
                "outside_monitored_history": [v for v in locations if not v["monitored_history"]]}
        finally:
            odb.close()
        after = fingerprint(path)
        rec["global_event_recheck_source_unchanged"] = before == after
        json_dump(summary_path, rec)


def audit_case(path, out, histories_only=False):
    from odbAccess import openOdb

    before = fingerprint(path)
    case_out = out / path.parent.parent.name / path.parent.name / path.stem
    case_out.mkdir(parents=True, exist_ok=True)
    rec = {"source": str(path), "before": before, "read_only": True}
    odb = openOdb(path=str(path), readOnly=True)
    try:
        step = odb.steps["Step-1"]
        rec["step_time_period"] = step.timePeriod
        rec["field_frames"] = len(step.frames)
        rec["last_field_time"] = step.frames[-1].frameValue
        load = step.historyRegions["Node ASSEMBLY.1"]
        histories = {key: list(load.historyOutputs[key].data) for key in ("U2", "RF2")}
        rec["load_histories"] = {key: {"count": len(vals), "first": vals[0], "last": vals[-1],
                                      "nonfinite_indices": [i for i, row in enumerate(vals) if not all(math.isfinite(v) for v in row)],
                                      "values": numeric_summary([v for t, v in vals])}
                                 for key, vals in histories.items()}
        rec["load_time_grids_equal"] = [t for t, v in histories["U2"]] == [t for t, v in histories["RF2"]]
        status, missing, status_events = {}, [], []
        for key, region in step.historyRegions.items():
            if "Element " not in key:
                continue
            if "STATUS" not in region.historyOutputs:
                missing.append({"region": key, "available": list(region.historyOutputs.keys())})
                continue
            data = list(region.historyOutputs["STATUS"].data)
            status[key] = data
            zeros = [(t, v) for t, v in data if v == 0]
            if zeros:
                first = zeros[0][0]
                prior = [(t, v) for t, v in data if t < first]
                status_events.append({"region": key, "time": first,
                                      "last_intact_time": prior[-1][0] if prior else None})
        rec["element_history_regions"] = len(status) + len(missing)
        rec["missing_STATUS_regions"] = missing
        rec["status_region_count"] = len(status)
        rec["status_lengths"] = sorted(set(len(v) for v in status.values()))
        rec["status_last_times"] = sorted(set(v[-1][0] for v in status.values() if v))
        rec["status_value_nonfinite"] = sum(sum(not math.isfinite(v) for t, v in data) for data in status.values())
        rec["status_events"] = sorted(status_events, key=lambda v: (v["time"], v["region"]))
        event_time = rec["status_events"][0]["time"] if status_events else None
        rec["first_monitored_deletion_time"] = event_time
        rec["field_check_scope"] = "none: histories only" if histories_only else "damage/STATUS scalars every frame; all variables last two frames"
        load_times = [t for t, v in histories["U2"]]
        status_aligned = all([t for t, v in vals] == load_times for vals in status.values())
        rec["status_load_time_grids_equal"] = status_aligned
        if status_aligned and rec["load_time_grids_equal"]:
            matrix = np.asarray([[v for t, v in status[key]] for key in status], dtype=float).T
            np.savez_compressed(str(case_out / "native_status.npz"), time=np.array(load_times),
                                STATUS=matrix, regions=np.array(list(status)))
            rows = [(t, u, histories["RF2"][i][1], int(np.count_nonzero(matrix[i] == 0)),
                     int(np.count_nonzero(np.isfinite(matrix[i]))), int(np.count_nonzero(~np.isfinite(matrix[i]))))
                    for i, (t, u) in enumerate(histories["U2"])]
            write_csv(case_out / "native_curve.csv", ["time_s", "U2_mm", "RF2_N", "observed_STATUS_zero", "known_STATUS", "unknown_STATUS"], rows)
            np.savez_compressed(str(case_out / "native_curve.npz"), time=np.array(load_times),
                                U2=np.array([v for t, v in histories["U2"]]),
                                RF2=np.array([v for t, v in histories["RF2"]]),
                                finite_load_mask=np.array([math.isfinite(row[1]) and math.isfinite(row[2]) for row in rows]))
            rec["last_history_samples"] = rows[-4:]
            zero_indices = [i for i in range(len(rows)) if rows[i][3] > 0]
            rec["first_monitored_deletion_index"] = zero_indices[0] if zero_indices else None
            rec["last_intact_history_index"] = zero_indices[0] - 1 if zero_indices and zero_indices[0] > 0 else None

        frame_rows, first_fields, field_error = [], {}, []
        for index in ([] if histories_only else range(len(step.frames))):
            frame = step.frames[index]
            row = {"index": index, "time": frame.frameValue}
            for name in ("STATUS", "SDEG", "DMICRT", "DUCTCRT", "PEEQ"):
                if name in frame.fieldOutputs:
                    field = frame.fieldOutputs[name]
                    try:
                        summary = field_summary(field)
                        row[name] = summary
                        if name != "STATUS" and summary["max"] is not None and summary["max"] > 0 and name not in first_fields:
                            first_fields[name] = {"index": index, "time": frame.frameValue,
                                                  "summary": summary,
                                                  "component_labels": list(field.componentLabels),
                                                  "locations": first_positive_locations(field)}
                    except Exception as error:
                        field_error.append({"index": index, "time": frame.frameValue, "variable": name, "error": str(error)})
            if index >= len(step.frames) - 2:
                row["all_field_finiteness"] = {}
                for name, field in frame.fieldOutputs.items():
                    try:
                        row["all_field_finiteness"][name] = field_summary(field)
                    except Exception as error:
                        row["all_field_finiteness"][name] = {"error": str(error)}
            frame_rows.append(row)
        rec["first_positive_fields"] = first_fields
        rec["field_errors"] = field_error
        rec["field_variables"] = list(step.frames[-1].fieldOutputs.keys())
        json_dump(case_out / "field_frames.json", frame_rows)
        rec["last_two_field_summaries"] = frame_rows[-2:]
        rec["energy_regions"] = []
        for key, region in step.historyRegions.items():
            if not all(name in region.historyOutputs for name in ("ALLKE", "ALLIE")):
                continue
            ke, ie = (list(region.historyOutputs[name].data) for name in ("ALLKE", "ALLIE"))
            same_time = [t for t, v in ke] == [t for t, v in ie]
            erec = {"region": key, "available": list(region.historyOutputs.keys()), "time_grids_equal": same_time}
            if same_time:
                rows = [(t, k, ie[i][1], k / ie[i][1] if ie[i][1] else float("nan")) for i, (t, k) in enumerate(ke)]
                write_csv(case_out / "native_energy.csv", ["time_s", "ALLKE_Nmm", "ALLIE_Nmm", "KE_over_IE"], rows)
                windows = [energy_window(rows, event_time, False, "strictly_before_first_monitored_deletion"),
                           energy_window(rows, event_time, True, "through_first_monitored_deletion")]
                for name in ("SDEG", "DMICRT", "DUCTCRT", "PEEQ"):
                    if name in first_fields:
                        windows.append(energy_window(rows, first_fields[name]["time"], False, "strictly_before_first_positive_" + name))
                erec.update({"count": len(rows), "last_samples": rows[-4:], "windows": windows})
            rec["energy_regions"].append(erec)
        rec["solver_diagnostics"] = {}
        for suffix in (".log", ".sta", ".msg"):
            companion = path.with_suffix(suffix)
            if companion.exists():
                text = companion.read_text(encoding="utf-8", errors="replace")
                selected = [line for line in text.splitlines() if any(word in line.lower() for word in
                            ("error", "warning", "fatal", "exceed", "complete", "wave speed", "deformation speed"))]
                rec["solver_diagnostics"][suffix] = {"sha256": fingerprint(companion)["sha256"],
                                                      "selected_lines": selected, "last_lines": text.splitlines()[-15:]}
    finally:
        odb.close()
    after = fingerprint(path)
    rec["after"] = after
    rec["source_unchanged"] = before == after
    json_dump(case_out / "summary.json", rec)
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--out", type=Path, default=Path("Z:/p1/sims/Ti/Validation/audit/history"))
    parser.add_argument("--case", action="append", help="case path relative to --root (repeatable)")
    parser.add_argument("--histories-only", action="store_true", help="omit field scans; preserve native history outputs")
    parser.add_argument("--density-campaign", action="store_true", help="all retained tri/kagome/hex fracture ODBs in the density campaign")
    parser.add_argument("--skip-existing", action="store_true", help="do not repeat cases with a completed summary")
    parser.add_argument("--summarize-only", action="store_true", help="derive saved event/windows and figure in standard Python; no ODB access")
    parser.add_argument("--refresh-event-locations", action="store_true", help="read global STATUS labels for saved detailed cases, without repeating other fields")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize_outputs(args.out)
        return
    if args.refresh_event_locations:
        refresh_global_event_locations(args.out)
        return
    cases = args.case or DEFAULT_CASES
    if args.density_campaign:
        with open(args.root / "case-index.csv", newline="", encoding="utf-8") as handle:
            cases = [r["odb"] for r in csv.DictReader(handle)
                     if r["campaign"] == "Density campaign" and r["mode"] == "Fracture"
                     and r["topology"] in ("tri", "kagome", "hex") and r["storage_status"] == "retained_odb"]
    errors = []
    for relative in cases:
        path = args.root / relative
        existing = args.out / path.parent.parent.name / path.parent.name / path.stem / "summary.json"
        if args.skip_existing and existing.exists():
            continue
        print("Auditing " + relative, flush=True)
        before_stat = path.stat()
        try:
            audit_case(path, args.out, histories_only=args.histories_only)
        except Exception as error:
            after_stat = path.stat()
            errors.append({"source": str(path), "error": str(error), "source_write_attempted": False,
                           "source_size_mtime_unchanged": (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)})
            print("Could not audit without upgrade: " + str(error), flush=True)
            continue
        print("Finished " + relative, flush=True)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.out.rglob("summary.json"))]
    json_dump(args.out / "native_history_audit.json", records)
    if errors:
        json_dump(args.out / "native_history_errors.json", errors)
    print("Source ODBs unchanged: " + str(all(v["source_unchanged"] for v in records)), flush=True)


if __name__ == "__main__":
    main()

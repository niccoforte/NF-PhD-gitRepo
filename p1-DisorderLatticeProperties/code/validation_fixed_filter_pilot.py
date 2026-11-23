"""Read-only 64-history pilot for the author-approved fixed Butterworth schedule.

Run only after Figure 10 promotion::
    python -B validation_fixed_filter_pilot.py --run

Writes one immutable report and CSV beneath Validation/audit/curve-processing;
never writes sources, Props, MLdata, final figures or manuscript files. FT peak
and whole-observed-history metrics are not initiation measurements. An optional
--events index must have its SHA256 supplied explicitly. It contains a list of
{topology, task, design, id, path, sha256, native_sha256, source_inp_sha256}
bindings to existing validation_failure_audit initiation.json receipts. Only
fully source-matched, undamaged local-strain brackets unlock endpoint load/work;
no fracture conversion modulus or production KJ is selected by this pilot.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re

import numpy as np

from resources.calculations import smooth
from validation_calculations import filter_force
from validation_curve_processing import (first_crossing, query_support_mask,
                                         sampled_odb_fingerprint, threshold_details)
from validation_stiffness import initial_work_stiffness, sha, _write_immutable_targets


ROOT = Path("Z:/p1/sims/Ti/Validation")
REGISTRY = Path(__file__).with_name("response_filter_cutoffs.json")
KEYS = ("topology", "task", "design", "id")


def identity(row):
    return tuple(row[k] if k != "id" else int(row[k]) for k in KEYS)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def metrics(x, y, task):
    """Recorded thresholds stay censored when the observed prefix never crosses."""
    peak = int(np.argmax(y))
    fits = {str(f): initial_work_stiffness(x, y, f) for f in (.3, .4, .5)}
    slopes = [row.get("estimate") for row in fits.values()]
    spread = (float(max(abs(np.asarray(slopes)/slopes[1]-1)))
              if all(v is not None and v > 0 for v in slopes) else None)
    result = dict(peak_response=float(y[peak]), peak_coordinate=float(x[peak]),
                  peak_index=peak, observed_work=float(np.trapz(y, x)),
                  initial_work_fits=fits, stiffness_window_relative_spread=spread)
    if task == "UT":
        ductility = threshold_details(x, y)
        end = first_crossing(y, .01)
        result.update(ductility25_observed=ductility["crossing_index"] is not None,
                      ductility25_coordinate=ductility["recorded_ductility"],
                      ductility25_index=ductility["crossing_index"],
                      work1_observed=end is not None, work1_index=end,
                      work1=float(np.trapz(y[:end+1], x[:end+1])) if end is not None else None)
    return result


def relative(value, reference):
    if value is None or reference is None or reference == 0:
        return None
    return float(100*(value/reference-1))


def material_event_index(record, time, x, y, event, odb_sha256):
    """Accept an existing local tensile-strain bracket, never STATUS or a peak."""
    if (not event.get("source_unchanged")
            or event.get("before", {}).get("sha256") != odb_sha256
            or event.get("after", {}).get("sha256") != odb_sha256
            or not event.get("threshold_found")
            or not event.get("undamaged_prefix_through_primary_lower_bound_verified")
            or event.get("nonfinite_before_threshold")
            or "surface tensile LE11 >=0.1105" not in event.get("criterion", "")
            or not float(event["prev_max_LE11"]) < .1105 <= float(event["cross_max_LE11"])):
        raise ValueError("Unverified native material-event source or undamaged strain bracket")
    lo, hi = event["prev_native_history"], event["cross_native_history"]
    i, j = int(lo["index"]), int(hi["index"])
    scales = record["normalisation_factors"]
    if not 0 < i < j < len(time):
        raise ValueError("Material bracket lies outside the observed native history")
    for point, index in ((lo, i), (hi, j)):
        if (abs(time[index]-float(point["time_s"])) > 2e-7
                or not np.isclose(x[index], float(point["U2_mm"])*scales[0], rtol=2e-6, atol=1e-7)
                or not np.isclose(y[index], float(point["RF2_N"])*scales[1], rtol=2e-6, atol=1e-5)):
            raise ValueError("Native event coordinates or times differ from the frozen curve")
    return i


def curve_report(record, time, x, y, cutoff, query_time, event_index=None):
    """Pure calculation on one genuine prefix; no resampling or curve export."""
    time, x, y = (np.asarray(v, dtype=float) for v in (time, x, y))
    fx, info = filter_force(time, x, cutoff)
    fy, _ = filter_force(time, y, cutoff)
    methods = {name: metrics(a, b, record["task"]) for name, a, b in
               (("raw", x, y), ("fixed", fx, fy),
                ("legacy_double_MA", x, np.asarray(smooth(smooth(y)))))}
    observed = query_support_mask(time, query_time)
    endpoint = {"status": "not_applicable_UT" if record["task"] == "UT" else
                "withheld_no_verified_native_material_event"}
    if record["task"] == "FT" and event_index is not None:
        i = int(event_index)
        if not 0 < i < len(time):
            raise ValueError("Material endpoint must be an observed noninitial sample")
        endpoint = dict(status="verified_native_material_event_force_opening_work_only",
                        native_index=i, time_s=float(time[i]),
                        KJ_status="withheld_final_conversion_and_work_policy_not_selected")
        px, prefix_info = filter_force(time[:i+1], x[:i+1], cutoff)
        py, _ = filter_force(time[:i+1], y[:i+1], cutoff)
        endpoint.update(filter_support="confirmed_native_intact_prefix_only",
                        fixed_prefix_filter=prefix_info,
                        area_interpretation="Provisional signed initial-triangle diagnostic from curve-derived k0, not independently qualified plastic work")
        for name, a, b in (("raw", x[:i+1], y[:i+1]), ("fixed", px, py)):
            fit = methods[name]["initial_work_fits"]["0.4"]
            work = float(np.trapz(b[:i+1], a[:i+1]))
            k0 = fit.get("estimate")
            usable = (k0 is not None and k0 > 0 and fit["end_index"] <= i
                      and np.all(np.diff(a[:i+1]) >= 0))
            area = work-float(b[i])**2/(2*k0) if usable else None
            endpoint[name] = dict(force_N=float(b[i]), opening_mm=float(a[i]), work_N_mm=work,
                                  curve_k0_N_per_mm=k0, signed_initial_triangle_area_N_mm=area,
                                  area_status="unresolved_initial_fit" if area is None else
                                  "nonnegative" if area >= 0 else "negative_unavailable")
        endpoint["fixed_minus_raw_percent"] = {
            key: relative(endpoint["fixed"][key], endpoint["raw"][key])
            for key in ("force_N", "opening_mm", "work_N_mm")}
    changes = {}
    for baseline in ("raw", "legacy_double_MA"):
        changes[baseline] = {key+"_percent": relative(methods["fixed"].get(key), methods[baseline].get(key))
                             for key in ("peak_response", "peak_coordinate", "observed_work",
                                         "ductility25_coordinate", "work1")}
        changes[baseline]["initial_work_slope_percent"] = relative(
            methods["fixed"]["initial_work_fits"]["0.4"].get("estimate"),
            methods[baseline]["initial_work_fits"]["0.4"].get("estimate"))
        for criterion in ("ductility25", "work1") if record["task"] == "UT" else ():
            changes[baseline][criterion+"_observation_changed"] = (
                methods["fixed"][criterion+"_observed"] != methods[baseline][criterion+"_observed"])
    return dict(**{k: record[k] for k in KEYS}, status="calculated_read_only",
                samples=len(time), last_observed_time_s=float(time[-1]), filter=info,
                planned_duration_s=float(query_time[-1]), planned_query_count=len(query_time),
                observed_query_count=int(observed.sum()), observed_query_mask=observed.tolist(),
                query_basis="201 original planned-time positions; no stretched response or target tensors",
                methods=methods, fixed_minus_baseline=changes, material_endpoint=endpoint)


def frozen_records(root, bind):
    folder = root/"audit/curve-processing"
    selection = read_json(bind(folder/"paired-examples/selection.json"))
    native = read_json(bind(folder/"paired-examples/native-index.json"))
    periodic = read_json(bind(folder/"periodic-cutoff-native/index.json"))
    expected = {identity(r): r for r in selection["records"]}
    if len(expected) != 56 or len(native) != 56 or len(selection["periodic"]) != 8:
        raise ValueError("Expected the frozen 56 paired disorder plus eight periodic records")
    for row in native:
        source = expected.get(identity(row))
        if source is None or any(row[k] != source[k] for k in ("source_inp", "source_odb", "transfer", "native")):
            raise ValueError("Native extraction differs from frozen paired selection")
    for row in selection["periodic"]:
        if row not in periodic:
            raise ValueError("Frozen periodic extraction differs from its index")
    records = selection["periodic"]+native
    keys = [identity(r) for r in records]
    if len(set(keys)) != 64 or {(r[0], r[1]) for r in keys} != {
            (lat, task) for lat in ("tri", "kagome", "hex", "FCC") for task in ("UT", "FT")}:
        raise ValueError("Duplicate or missing topology/test members")
    for lat in ("tri", "kagome", "hex", "FCC"):
        for task in ("UT", "FT"):
            family = [r for r in records if r["topology"] == lat and r["task"] == task]
            if len(family) != 8 or sum(r["design"] == "periodic" for r in family) != 1:
                raise ValueError("Each family requires one periodic and seven paired disorder cases")
    return records


def planned_time(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    steps = re.findall(r"^\*Dynamic,\s*Explicit[^\n]*\n([^*]+)", text, re.I | re.M)
    if len(steps) != 1:
        raise ValueError("One original Explicit loading schedule required")
    end = float(steps[0].splitlines()[0].split(",")[1])
    if not np.isfinite(end) or end <= 0:
        raise ValueError("Positive planned simulation duration required")
    return np.linspace(0., end, 201)


def run(root=ROOT, out=None, events=None, events_sha256=None, replace_report_sha256=None):
    root = Path(root).resolve()
    out = Path(out).resolve() if out else root/"audit/curve-processing/fixed-filter-pilot"
    if not out.is_relative_to(root/"audit/curve-processing"):
        raise ValueError("Pilot outputs must remain beneath the existing curve-processing audit")
    replaced = None
    if replace_report_sha256 is not None:
        prior, table = out/"pilot.json", out/"metrics.csv"
        if sha(prior) != replace_report_sha256 or read_json(prior).get("schema") != "fixed_filter_pilot_v1":
            raise ValueError("Replacement requires the exact existing unpromoted pilot report hash")
        replaced = {str(prior): replace_report_sha256, str(table): sha(table)}
    bindings = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Frozen source fingerprint differs: "+str(path))
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError("Source changed during pilot: "+str(path))
        bindings[str(path)] = digest
        return path
    for implementation in (__file__, filter_force.__code__.co_filename,
                           initial_work_stiffness.__code__.co_filename,
                           threshold_details.__code__.co_filename, smooth.__code__.co_filename):
        bind(implementation)
    registry = read_json(bind(REGISTRY))
    if any(registry["filter"][key] != value for key, value in
           (("order", 2), ("family", "Butterworth"), ("kind", "lowpass"))):
        raise ValueError("Cutoff registry requires another filter implementation")
    records = frozen_records(root, bind)
    material = {}
    if events is not None:
        if not events_sha256:
            raise ValueError("An explicit SHA256 must bind the optional material-event index")
        for entry in read_json(bind(events, events_sha256)):
            if identity(entry) in material:
                raise ValueError("Duplicate material-event binding")
            material[identity(entry)] = entry
        available = {identity(r) for r in records if r["task"] == "FT"}
        if not set(material).issubset(available):
            raise ValueError("Material-event index names an unfrozen or non-FT case")
    odb_checks, rows = {}, []
    for record in records:
        inp = bind(record["source_inp"], record["source_inp_sha256"])
        bind(record["transfer"], record["transfer_sha256"])
        native = bind(record["native"], record["native_sha256"])
        odb = Path(record["source_odb"])
        if "source_odb_sha256" in record:
            bind(odb, record["source_odb_sha256"])
            proof = {"full_file_sha256": record["source_odb_sha256"]}
        else:
            proof = sampled_odb_fingerprint(odb)
            if proof != record["source_odb_fingerprint"]:
                raise ValueError("Recorded archive ODB fingerprint differs: "+str(odb))
            odb_checks[str(odb)] = proof
        if not record.get("source_unchanged") or not record.get("observed_support_only"):
            raise ValueError("Native extraction does not attest observed source support")
        with np.load(native, allow_pickle=False) as data:
            time, x, y = (data[k] for k in ("time", "x", "y"))
        if len(time) != record["samples"]:
            raise ValueError("Native sample count differs from its extraction")
        event_index, event_binding, event_issue = None, None, None
        if identity(record) in material:
            entry = material[identity(record)]
            if (entry["native_sha256"] != record["native_sha256"]
                    or entry["source_inp_sha256"] != record["source_inp_sha256"]):
                raise ValueError("Material event is bound to another curve or input")
            event_path = bind(entry["path"], entry["sha256"])
            event = read_json(event_path)
            odb_sha = sha(odb)
            bind(odb, odb_sha)
            if Path(event["source"]).resolve() != odb.resolve():
                raise ValueError("Material receipt names another solver source")
            event_binding = {"path": str(event_path), "sha256": entry["sha256"]}
            if entry.get("status", event.get("status")) not in (None, "qualified_observed_event"):
                event_issue = dict(status="withheld_unqualified_or_censored_native_material_event",
                                   collector_status=entry.get("status", event.get("status")),
                                   reasons=event.get("nonfinite_before_threshold", []))
            else:
                try:
                    event_index = material_event_index(record, time, x, y, event, odb_sha)
                except (ValueError, KeyError, TypeError) as error:
                    event_issue = dict(status="withheld_invalid_material_event_bracket", reason=str(error))
        try:
            row = curve_report(record, time, x, y,
                               float(registry["cutoffs_hz"][record["topology"]][record["task"]]),
                               planned_time(inp), event_index)
        except ValueError as error:
            row = dict(**{k: record[k] for k in KEYS}, status="unavailable_sampling_or_response",
                       reason=str(error), material_endpoint={"status": "withheld_invalid_native_support"})
        row.update(native=str(native), native_sha256=record["native_sha256"],
                   source_inp=str(inp), source_odb=str(odb), source_odb_identity=proof,
                   event_binding=event_binding)
        if event_issue is not None:
            row["material_endpoint"] = event_issue
        rows.append(row)
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("Protected pilot input changed: "+path)
    for path, proof in odb_checks.items():
        if sampled_odb_fingerprint(path) != proof:
            raise ValueError("Archive source changed during read-only pilot: "+path)
    report = dict(schema="fixed_filter_pilot_v1", scope="Frozen 64 histories, 10mm/20%, original periodic and paired IDs",
                  replaced_unpromoted_pilot=replaced,
                  cutoff_registry=str(REGISTRY), cutoff_registry_sha256=bindings[str(REGISTRY.resolve())],
                  source_bindings=bindings, records=rows, native_histories=len(rows),
                  calculated=sum(r["status"] == "calculated_read_only" for r in rows),
                  FT_event_mechanics_available=sum(r["material_endpoint"]["status"].startswith("verified_") for r in rows),
                  production_writes=False, new_solver_jobs=0,
                  limitations=["Array and INP/transfer hashes are full-file; disordered ODB identities retain their explicitly sampled fingerprint level unless a full-hash event receipt is supplied.",
                               "Whole-prefix FT response metrics and force maxima are not material initiation measurements.",
                               "Window spread and old-method differences are sensitivities, not absolute physical error or confidence intervals.",
                               "No missing thresholds, strain tails, material events, modulus, toughness or mask-aware ML tensors are manufactured."])
    flat = []
    for row in rows:
        item = {k: row[k] for k in KEYS+("status",)}
        if row["status"] == "calculated_read_only":
            fixed = row["methods"]["fixed"]
            item.update(cutoff_Hz=row["filter"]["cutoff_Hz"], samples=row["samples"],
                        observed_query_count=row["observed_query_count"], peak=fixed["peak_response"],
                        initial_work_slope=fixed["initial_work_fits"]["0.4"].get("estimate"),
                        window_spread=fixed["stiffness_window_relative_spread"],
                        ductility25=fixed.get("ductility25_coordinate"), work1=fixed.get("work1"))
        item["material_endpoint_status"] = row["material_endpoint"]["status"]
        flat.append(item)
    stream = io.StringIO(newline="")
    columns = list(dict.fromkeys(k for row in flat for k in row))
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(flat)
    targets = {out/"pilot.json": json.dumps(report, indent=2, allow_nan=False)+"\n",
               out/"metrics.csv": stream.getvalue()}
    if replaced is None:
        _write_immutable_targets(targets)
    else:
        if any(sha(path) != digest for path, digest in replaced.items()):
            raise ValueError("Staged pilot changed during reviewed replacement")
        for path, text in targets.items():
            path.write_text(text, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the read-only staged pilot after Figure 10 promotion")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--events-sha256")
    parser.add_argument("--replace-report-sha256", help="Replace only the exact prior unpromoted pilot.json and its companion table")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run explicitly after Figure 10 promotion; --help never runs the pilot")
    result = run(args.root, args.out, args.events, args.events_sha256, args.replace_report_sha256)
    print(json.dumps({k: result[k] for k in ("native_histories", "calculated", "FT_event_mechanics_available")}))

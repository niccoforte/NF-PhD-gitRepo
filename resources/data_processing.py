import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from resources.calculations import (
    get_ductileData,
    get_fractureData,
    get_frequencies,
    get_nodes,
    get_struts,
)


### Node input processing
def _sim_num_from_filename(ffile):
    if "-per-" in ffile:
        return 0
    return int(ffile.split("-")[-1][:-4])

def _dis_enabled(dat, dis):
    return dat.dis.lower() == dis.lower() or dat.dis.lower() == "both"

def _rows_to_indexed_df(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows, index=[int(i[0]) for i in rows])
    df = df.drop(0, axis=1).sort_index()
    df.columns = range(df.columns.size)
    return df

def _write_df_if_nonempty(df, csv_path):
    if df is None or df.empty or df.shape[1] == 0:
        return None
    df.to_csv(csv_path)
    return df

def _append_frequency_row(rows, csv_path, num, dat):
    if not dat.freq:
        return
    freqs = get_frequencies(csv_path)
    if len(freqs) > 0:
        rows.append(np.insert(freqs, 0, num))

def create_inputCSV(directory, dat):
    duct_disNodes_n = []
    duct_disNodes_f = []
    duct_disStruts_s = []
    duct_disStruts_f = []
    frac_disNodes_n = []
    frac_disNodes_f = []
    frac_disStruts_s = []
    frac_disStruts_f = []

    path_raw = directory + "transfer/"
    for ffile in os.listdir(path_raw):
        if not ffile.endswith(".csv"):
            continue
        try:
            num = _sim_num_from_filename(ffile)
        except ValueError:
            continue

        if "per" in ffile and "Ductile" in ffile and "IN-n" in ffile:
            nodes, nodesCoords, nodes_df = get_nodes(path_raw + ffile)
            duct_disNodes_n.insert(0, np.insert(nodes_df.to_numpy().flatten(), 0, 0))
        elif "per" in ffile and "Ductile" in ffile and "IN-s" in ffile:
            thicks = get_struts(path_raw + ffile)
            duct_disStruts_s.insert(0, np.insert(thicks, 0, 0))
        elif "per" in ffile and "Fracture" in ffile and "IN-n" in ffile:
            nodes, nodesCoords, nodes_df = get_nodes(path_raw + ffile)
            frac_disNodes_n.insert(0, np.insert(nodes_df.to_numpy().flatten(), 0, 0))
        elif "per" in ffile and "Fracture" in ffile and "IN-s" in ffile:
            thicks = get_struts(path_raw + ffile)
            frac_disStruts_s.insert(0, np.insert(thicks, 0, 0))
        elif "disNodes" in ffile and "Ductile" in ffile and "IN-n" in ffile:
            nodes, nodesCoords, nodes_df = get_nodes(path_raw + ffile)
            duct_disNodes_n.append(np.insert(nodes_df.to_numpy().flatten(), 0, num))
        elif "disNodes" in ffile and "Ductile" in ffile and "IN-f" in ffile:
            _append_frequency_row(duct_disNodes_f, path_raw + ffile, num, dat)
        elif "disStruts" in ffile and "Ductile" in ffile and "IN-s" in ffile:
            thicks = get_struts(path_raw + ffile)
            duct_disStruts_s.append(np.insert(thicks, 0, num))
        elif "disStruts" in ffile and "Ductile" in ffile and "IN-f" in ffile:
            _append_frequency_row(duct_disStruts_f, path_raw + ffile, num, dat)
        elif "disNodes" in ffile and "Fracture" in ffile and "IN-n" in ffile:
            nodes, nodesCoords, nodes_df = get_nodes(path_raw + ffile)
            frac_disNodes_n.append(np.insert(nodes_df.to_numpy().flatten(), 0, num))
        elif "disNodes" in ffile and "Fracture" in ffile and "IN-f" in ffile:
            _append_frequency_row(frac_disNodes_f, path_raw + ffile, num, dat)
        elif "disStruts" in ffile and "Fracture" in ffile and "IN-s" in ffile:
            thicks = get_struts(path_raw + ffile)
            frac_disStruts_s.append(np.insert(thicks, 0, num))
        elif "disStruts" in ffile and "Fracture" in ffile and "IN-f" in ffile:
            _append_frequency_row(frac_disStruts_f, path_raw + ffile, num, dat)

    UTdisNodesINn_df = None
    UTdisNodesINf_df = None
    UTdisStrutsINs_df = None
    UTdisStrutsINf_df = None
    FTdisNodesINn_df = None
    FTdisNodesINf_df = None
    FTdisStrutsINs_df = None
    FTdisStrutsINf_df = None

    if dat.UTmechTest:
        if _dis_enabled(dat, "disNodes"):
            UTdisNodesINn_df = _write_df_if_nonempty(_rows_to_indexed_df(duct_disNodes_n), directory + "Ductile-disNodes-IN.csv")
            UTdisNodesINf_df = _write_df_if_nonempty(_rows_to_indexed_df(duct_disNodes_f), directory + "Ductile-disNodes-INf.csv")
        if _dis_enabled(dat, "disStruts"):
            UTdisStrutsINs_df = _write_df_if_nonempty(_rows_to_indexed_df(duct_disStruts_s), directory + "Ductile-disStruts-IN.csv")
            UTdisStrutsINf_df = _write_df_if_nonempty(_rows_to_indexed_df(duct_disStruts_f), directory + "Ductile-disStruts-INf.csv")

    if dat.FTmechTest:
        if _dis_enabled(dat, "disNodes"):
            FTdisNodesINn_df = _write_df_if_nonempty(_rows_to_indexed_df(frac_disNodes_n), directory + "Fracture-disNodes-IN.csv")
            FTdisNodesINf_df = _write_df_if_nonempty(_rows_to_indexed_df(frac_disNodes_f), directory + "Fracture-disNodes-INf.csv")
        if _dis_enabled(dat, "disStruts"):
            FTdisStrutsINs_df = _write_df_if_nonempty(_rows_to_indexed_df(frac_disStruts_s), directory + "Fracture-disStruts-IN.csv")
            FTdisStrutsINf_df = _write_df_if_nonempty(_rows_to_indexed_df(frac_disStruts_f), directory + "Fracture-disStruts-INf.csv")

    return (
        UTdisNodesINn_df,
        UTdisNodesINf_df,
        UTdisStrutsINs_df,
        UTdisStrutsINf_df,
        FTdisNodesINn_df,
        FTdisNodesINf_df,
        FTdisStrutsINs_df,
        FTdisStrutsINf_df,
    )


### Curve output processing
def _raw_files_by_num(directory, mech_test, dis, token):
    path_raw = directory + "transfer/"
    raw_files = {}
    if not os.path.isdir(path_raw):
        return raw_files

    for ffile in os.listdir(path_raw):
        if not ffile.endswith(".csv"):
            continue
        if mech_test not in ffile or token not in ffile:
            continue
        if dis not in ffile and "-per-" not in ffile:
            continue
        try:
            raw_files[_sim_num_from_filename(ffile)] = ffile
        except ValueError:
            pass
    return raw_files

def _read_final_ids(csv_path):
    if csv_path is None or not os.path.exists(csv_path):
        return set()
    idx = pd.read_csv(csv_path, usecols=[0]).iloc[:, 0]
    final_ids = set()
    for val in idx:
        try:
            final_ids.add(int(float(val)))
        except (TypeError, ValueError):
            pass
    return final_ids

def _output_row_info(output_rows):
    info = {}
    for row in output_rows:
        sim_id = int(row[0])
        row_has_nan = bool(pd.isna(row).any())
        failure_idx = np.nan
        if len(row) > 1 and not pd.isna(row[1]):
            failure_idx = float(row[1])

        entry = info.setdefault(
            sim_id,
            {
                "output_failure_idx": np.nan,
                "output_has_nan": False,
                "output_row_count": 0,
            },
        )
        entry["output_row_count"] += 1
        entry["output_has_nan"] = entry["output_has_nan"] or row_has_nan
        if not pd.isna(failure_idx):
            entry["output_failure_idx"] = failure_idx
    return info

def _output_drop_ids(output_info):
    drop_ids = []
    for sim_id, info in output_info.items():
        failure_idx = info["output_failure_idx"]
        failure_idx_zero = sim_id != 0 and not pd.isna(failure_idx) and int(failure_idx) == 0
        if info["output_has_nan"] or failure_idx_zero:
            drop_ids.append(sim_id)
    return sorted(set(drop_ids))

def _write_manifest(directory, dat, mech_test, mech_mode, dis, input_token, input_csv, output_csv, freq_csv, output_info):
    raw_input_files = _raw_files_by_num(directory, mech_test, dis, input_token)
    raw_output_files = _raw_files_by_num(directory, mech_test, dis, "OUT-")
    raw_freq_files = _raw_files_by_num(directory, mech_test, dis, "IN-f")

    final_input_ids = _read_final_ids(input_csv)
    final_output_ids = _read_final_ids(output_csv)
    final_freq_ids = _read_final_ids(freq_csv) if dat.freq else set()

    sim_ids = sorted(
        set(raw_input_files)
        | set(raw_output_files)
        | set(raw_freq_files)
        | final_input_ids
        | final_output_ids
        | final_freq_ids
        | set(output_info)
    )

    rows = []
    for sim_id in sim_ids:
        has_input = sim_id in raw_input_files
        has_output = sim_id in raw_output_files
        has_freq = sim_id in raw_freq_files
        input_in_final = sim_id in final_input_ids
        output_in_final = sim_id in final_output_ids
        freq_in_final = sim_id in final_freq_ids
        included = input_in_final and output_in_final
        info = output_info.get(sim_id, {})
        failure_idx = info.get("output_failure_idx", np.nan)
        output_has_nan = bool(info.get("output_has_nan", False))
        failure_idx_zero = sim_id != 0 and not pd.isna(failure_idx) and int(failure_idx) == 0

        reasons = []
        if sim_id == 0:
            reasons.append("periodic_reference")
        if sim_id != 0 and not has_input:
            reasons.append("missing_input")
        if sim_id != 0 and not has_output:
            reasons.append("missing_output")
        if output_has_nan:
            reasons.append("output_nan")
        if failure_idx_zero:
            reasons.append("failure_idx_zero")
        if has_freq and not dat.freq:
            reasons.append("freq_input_ignored_nonfreq_workflow")
        if dat.freq and sim_id != 0 and not has_freq:
            reasons.append("missing_freq_input")
        if dat.freq and has_freq and not freq_in_final:
            reasons.append("freq_input_empty_or_dropped")
        if sim_id != 0 and has_input and has_output and not included and not reasons:
            reasons.append("filtered_after_processing")

        rows.append(
            {
                "sim_id": sim_id,
                "mech_mode": mech_mode,
                "mech_test": mech_test,
                "disorder": dis,
                "input_token": input_token,
                "frequency_workflow": bool(dat.freq),
                "has_input": has_input,
                "has_output": has_output,
                "has_freq_input": has_freq,
                "input_in_final_csv": input_in_final,
                "output_in_final_csv": output_in_final,
                "freq_in_final_csv": freq_in_final,
                "included": included,
                "output_failure_idx": failure_idx,
                "output_has_nan": output_has_nan,
                "output_row_count": int(info.get("output_row_count", 0)),
                "reason": ";".join(reasons) if reasons else "included",
                "input_file": raw_input_files.get(sim_id, ""),
                "output_file": raw_output_files.get(sim_id, ""),
                "freq_file": raw_freq_files.get(sim_id, ""),
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(directory + f"{mech_test}-{dis}-manifest.csv", index=False)
    return manifest_df

def _process_output_dataset(directory, dat, mech_test, mech_mode, dis, input_token, input_csv, output_csv, freq_csv, output_rows):
    output_info = _output_row_info(output_rows)
    output_drop_ids = _output_drop_ids(output_info)
    raw_output_ids = set(_raw_files_by_num(directory, mech_test, dis, "OUT-"))

    output_df = None
    if output_rows:
        output_df = pd.DataFrame(output_rows, index=[int(i[0]) for i in output_rows])
        output_df = output_df.drop(output_drop_ids, axis=0, errors="ignore").drop(0, axis=1).sort_index()
        output_df.columns = range(output_df.columns.size)
        output_df.to_csv(output_csv)

    input_df = None
    if os.path.exists(input_csv):
        input_df = pd.read_csv(input_csv, index_col=0)
        missing_output_ids = [int(i) for i in input_df.index if int(i) not in raw_output_ids]
        input_drop_ids = sorted(set(output_drop_ids + missing_output_ids))
        input_df = input_df.drop(input_drop_ids, axis=0, errors="ignore").sort_index()
        input_df.columns = range(input_df.columns.size)
        input_df.to_csv(input_csv)

    freq_df = None
    if dat.freq and freq_csv is not None and os.path.exists(freq_csv):
        freq_df = pd.read_csv(freq_csv, index_col=0)
        if freq_df.shape[1] > 0:
            missing_output_ids = [int(i) for i in freq_df.index if int(i) not in raw_output_ids]
            freq_drop_ids = sorted(set(output_drop_ids + missing_output_ids))
            freq_df = freq_df.drop(freq_drop_ids, axis=0, errors="ignore").sort_index()
            freq_df.columns = range(freq_df.columns.size)
            if len(freq_df) > 0:
                freq_df.to_csv(freq_csv)
            else:
                freq_df = None
        else:
            freq_df = None

    _write_manifest(directory, dat, mech_test, mech_mode, dis, input_token, input_csv, output_csv, freq_csv, output_info)
    return output_df, input_df, freq_df

def create_outputCSV(directory, dat):
    duct_disNodes = []
    duct_disStruts = []
    frac_disNodes = []
    frac_disStruts = []

    path_raw = directory + "transfer/"
    for ffile in os.listdir(path_raw):
        if not ffile.endswith(".csv"):
            continue
        try:
            num = _sim_num_from_filename(ffile)
        except ValueError:
            continue

        if "per" in ffile and "Ductile" in ffile and "OUT-" in ffile:
            output_df = get_ductileData(path_raw + ffile, crit=0.25)
            duct_disNodes.insert(0, np.insert(output_df.x.tolist(), 0, 0))
            duct_disStruts.insert(0, np.insert(output_df.x.tolist(), 0, 0))
            duct_disNodes.insert(1, np.insert(output_df.y_sm.tolist(), 0, 0))
            duct_disStruts.insert(1, np.insert(output_df.y_sm.tolist(), 0, 0))
        elif "per" in ffile and "Fracture" in ffile and "OUT-" in ffile:
            output_df = get_fractureData(path_raw + ffile)
            frac_disNodes.insert(0, np.insert(output_df.x.tolist(), 0, 0))
            frac_disStruts.insert(0, np.insert(output_df.x.tolist(), 0, 0))
            frac_disNodes.insert(1, np.insert(output_df.y_sm.tolist(), 0, 0))
            frac_disStruts.insert(1, np.insert(output_df.y_sm.tolist(), 0, 0))
        elif "disNodes" in ffile and "Ductile" in ffile and "OUT-" in ffile:
            output_df = get_ductileData(path_raw + ffile, crit=0.25)
            duct_disNodes.append(np.insert(output_df.y_sm.tolist(), 0, num))
        elif "disStruts" in ffile and "Ductile" in ffile and "OUT-" in ffile:
            output_df = get_ductileData(path_raw + ffile, crit=0.25)
            duct_disStruts.append(np.insert(output_df.y_sm.tolist(), 0, num))
        elif "disNodes" in ffile and "Fracture" in ffile and "OUT-" in ffile:
            output_df = get_fractureData(path_raw + ffile)
            frac_disNodes.append(np.insert(output_df.y_sm.tolist(), 0, num))
        elif "disStruts" in ffile and "Fracture" in ffile and "OUT-" in ffile:
            output_df = get_fractureData(path_raw + ffile)
            frac_disStruts.append(np.insert(output_df.y_sm.tolist(), 0, num))

    UTdisNodesINn_df = None
    UTdisNodesINf_df = None
    UTdisStrutsINs_df = None
    UTdisStrutsINf_df = None
    UTdisNodesOUT_df = None
    UTdisStrutsOUT_df = None
    FTdisNodesINn_df = None
    FTdisNodesINf_df = None
    FTdisStrutsINs_df = None
    FTdisStrutsINf_df = None
    FTdisNodesOUT_df = None
    FTdisStrutsOUT_df = None

    if dat.UTmechTest:
        if _dis_enabled(dat, "disNodes"):
            UTdisNodesOUT_df, UTdisNodesINn_df, UTdisNodesINf_df = _process_output_dataset(
                directory,
                dat,
                "Ductile",
                "UT",
                "disNodes",
                "IN-n",
                directory + "Ductile-disNodes-IN.csv",
                directory + "Ductile-disNodes-OUT.csv",
                directory + "Ductile-disNodes-INf.csv",
                duct_disNodes,
            )
        if _dis_enabled(dat, "disStruts"):
            UTdisStrutsOUT_df, UTdisStrutsINs_df, UTdisStrutsINf_df = _process_output_dataset(
                directory,
                dat,
                "Ductile",
                "UT",
                "disStruts",
                "IN-s",
                directory + "Ductile-disStruts-IN.csv",
                directory + "Ductile-disStruts-OUT.csv",
                directory + "Ductile-disStruts-INf.csv",
                duct_disStruts,
            )

    if dat.FTmechTest:
        if _dis_enabled(dat, "disNodes"):
            FTdisNodesOUT_df, FTdisNodesINn_df, FTdisNodesINf_df = _process_output_dataset(
                directory,
                dat,
                "Fracture",
                "FT",
                "disNodes",
                "IN-n",
                directory + "Fracture-disNodes-IN.csv",
                directory + "Fracture-disNodes-OUT.csv",
                directory + "Fracture-disNodes-INf.csv",
                frac_disNodes,
            )
        if _dis_enabled(dat, "disStruts"):
            FTdisStrutsOUT_df, FTdisStrutsINs_df, FTdisStrutsINf_df = _process_output_dataset(
                directory,
                dat,
                "Fracture",
                "FT",
                "disStruts",
                "IN-s",
                directory + "Fracture-disStruts-IN.csv",
                directory + "Fracture-disStruts-OUT.csv",
                directory + "Fracture-disStruts-INf.csv",
                frac_disStruts,
            )

    return (
        UTdisNodesOUT_df,
        UTdisStrutsOUT_df,
        FTdisNodesOUT_df,
        FTdisStrutsOUT_df,
        UTdisNodesINn_df,
        UTdisNodesINf_df,
        UTdisStrutsINs_df,
        UTdisStrutsINf_df,
        FTdisNodesINn_df,
        FTdisNodesINf_df,
        FTdisStrutsINs_df,
        FTdisStrutsINf_df,
    )


### Field-output processing
def _as_path(directory):
    return Path(directory)

def _field_root(directory, field_dir=""):
    path = Path(field_dir)
    if not path.is_absolute():
        path = _as_path(directory) / path
    return path

def _field_raw_root(directory, field_dir="", raw_dir="transfer"):
    path = Path(raw_dir)
    if not path.is_absolute():
        path = _field_root(directory, field_dir) / path
    return path

def _field_stem_from_path(path, field_prefix="FIELDu-"):
    stem = path.stem
    for prefix in [field_prefix, "FIELDu-", "FIELD-"]:
        if prefix and stem.startswith(prefix):
            return stem[len(prefix):]
    return stem

def _npz_first_string(npz, keys, default=""):
    for key in keys:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.size == 0:
                return default
            return str(arr.reshape(-1)[0])
    return default

def _npz_string_list(npz, keys):
    for key in keys:
        if key in npz.files:
            return [str(item) for item in np.asarray(npz[key]).reshape(-1)]
    return []

def _coerce_sample_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        fval = float(text)
        if np.isfinite(fval) and fval.is_integer():
            return int(fval)
    except (TypeError, ValueError):
        pass
    return text

def _field_sample_id(npz, stem=None):
    if stem is not None and "-per-" in str(stem).lower():
        return 0
    sample_id = _coerce_sample_id(_npz_first_string(npz, ["sample_id", "sample_ids", "ids", "indices", "index"]))
    if isinstance(sample_id, int):
        return sample_id
    sample_number = _coerce_sample_id(_npz_first_string(npz, ["sample_number", "sample_numbers"]))
    if sample_number is not None:
        return sample_number
    return sample_id

def _field_mode_from_stem(stem):
    low = str(stem).lower()
    if low.startswith("ductile-"):
        return "UT", "Ductile"
    if low.startswith("fracture-"):
        return "FT", "Fracture"
    return None, None

def _field_mode_from_npz(npz, stem):
    mode = _npz_first_string(npz, ["mode"], default="")
    mode = mode.upper()
    if mode in ["UT", "FT"]:
        return mode, "Ductile" if mode == "UT" else "Fracture"
    return _field_mode_from_stem(stem)

def _field_raw_files(directory, field_dir="", raw_dir="transfer", field_prefix="FIELDu-"):
    root = _field_raw_root(directory, field_dir=field_dir, raw_dir=raw_dir)
    if not root.exists():
        return []
    files = sorted(root.glob(f"{field_prefix}*.npz"))
    if not files and field_prefix == "FIELDu-":
        files = sorted(root.glob("FIELD-*.npz"))
    return files

def _field_family_name(field_prefix="FIELDu-"):
    return str(field_prefix).strip("-") or "FIELDu"

def _field_values_to_fnc(values):
    values = np.asarray(values)
    if values.ndim == 3:
        return values
    if values.ndim == 4 and values.shape[0] == 1:
        return values[0]
    raise ValueError(f"Raw field values must have shape [frame,node,component], got {values.shape}.")

def create_fieldIndexCSV(directory, field_dir="", raw_dir="transfer", index_name=None, field_prefix="FIELDu-", dat=None, dis=None):
    """Create a lightweight audit table for raw FIELDu-*.npz outputs."""
    dis = dis or (dat.dis if dat is not None else "disNodes")
    field_family = _field_family_name(field_prefix)
    rows = []
    for path in _field_raw_files(directory, field_dir=field_dir, raw_dir=raw_dir, field_prefix=field_prefix):
        with np.load(path, allow_pickle=True) as npz:
            values = _field_values_to_fnc(npz["Y"])
            valid = np.asarray(npz["valid_mask"], dtype=bool) if "valid_mask" in npz.files else np.isfinite(values)
            stem = _npz_first_string(npz, ["sample_stem", "sample_stems"], default=_field_stem_from_path(path, field_prefix))
            mode, mech_test = _field_mode_from_npz(npz, stem)
            finite = np.isfinite(values)
            rows.append(
                {
                    "sample_id": _field_sample_id(npz, stem),
                    "sample_number": _npz_first_string(npz, ["sample_number", "sample_numbers"]),
                    "stem": stem,
                    "mech_mode": mode,
                    "mech_test": mech_test,
                    "n_frames": int(values.shape[0]),
                    "n_nodes": int(values.shape[1]),
                    "n_components": int(values.shape[2]),
                    "components": ",".join(_npz_string_list(npz, ["components", "component_names", "variables"])),
                    "valid_fraction": float(valid.mean()) if valid.size else np.nan,
                    "value_min": float(np.nanmin(values)) if np.any(finite) else np.nan,
                    "value_max": float(np.nanmax(values)) if np.any(finite) else np.nan,
                    "field_file": str(path),
                    "source_odb_path": _npz_first_string(npz, ["source_odb_path", "source_odb_paths"]),
                    "source_inp_path": _npz_first_string(npz, ["source_inp_path", "source_inp_paths"]),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mech_mode", "sample_id", "stem"], kind="stable").reset_index(drop=True)
    out_root = _field_root(directory, field_dir=field_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if index_name is not None:
        df.to_csv(out_root / index_name, index=False)
    elif not df.empty:
        for mech_test, mode_df in df.groupby("mech_test", sort=False):
            if pd.isna(mech_test) or str(mech_test) == "":
                continue
            mode_df.to_csv(out_root / f"{mech_test}-{dis}-{field_family}-index.csv", index=False)
    return df

def _field_sort_key(item):
    sample_id = item["sample_id"]
    if isinstance(sample_id, (int, np.integer)):
        return (0, int(sample_id))
    try:
        return (0, int(float(sample_id)))
    except (TypeError, ValueError):
        return (1, str(sample_id))

def _stack_field_group(items, out_path, mode, mech_test, dis, dtype="float32"):
    if not items:
        return None

    items = sorted(items, key=_field_sort_key)
    sample_numbers = [item["sample_id"] for item in items]
    id_counts = {}
    for sample_id in sample_numbers:
        id_counts[str(sample_id)] = id_counts.get(str(sample_id), 0) + 1
    sample_ids = [
        f"{item['sample_id']}::{item['stem']}" if id_counts[str(item["sample_id"])] > 1 else item["sample_id"]
        for item in items
    ]

    loaded = []
    max_frames = 0
    n_nodes = None
    components = None
    field_names = None
    frame_values = None
    node_labels = None
    node_coords0 = None
    elements = None

    for item in items:
        with np.load(item["path"], allow_pickle=True) as npz:
            values = _field_values_to_fnc(np.asarray(npz["Y"], dtype=dtype))
            valid = _field_values_to_fnc(np.asarray(npz["valid_mask"], dtype=bool)) if "valid_mask" in npz.files else np.isfinite(values)
            current_components = _npz_string_list(npz, ["components", "component_names", "variables"])
            current_field_names = _npz_string_list(npz, ["field_names"])
            current_labels = np.asarray(npz["node_labels"], dtype=int) if "node_labels" in npz.files else np.arange(values.shape[1]) + 1
            current_coords = np.asarray(npz["node_coords"], dtype=dtype) if "node_coords" in npz.files else None
            current_frames = np.asarray(npz["frame_values"], dtype=float) if "frame_values" in npz.files else np.arange(values.shape[0])
            current_elements = np.asarray(npz["elements"], dtype=int) if "elements" in npz.files else np.empty((0, 3), dtype=int)
            stem = _npz_first_string(npz, ["sample_stem", "sample_stems"], default=_field_stem_from_path(item["path"]))

        if n_nodes is None:
            n_nodes = values.shape[1]
            node_labels = current_labels
            node_coords0 = current_coords
            components = current_components if current_components else [f"c{i}" for i in range(values.shape[2])]
            field_names = current_field_names
            frame_values = current_frames
            elements = current_elements
        else:
            if values.shape[1] != n_nodes:
                raise ValueError(f"{mode}: {stem} has {values.shape[1]} nodes, expected {n_nodes}.")
            if current_components and list(current_components) != list(components):
                raise ValueError(f"{mode}: {stem} components {current_components} do not match {components}.")
            if len(current_labels) == len(node_labels) and not np.array_equal(current_labels, node_labels):
                raise ValueError(f"{mode}: {stem} node labels do not match the first sample.")

        if item["sample_id"] == 0 and current_coords is not None:
            node_coords0 = current_coords
        max_frames = max(max_frames, values.shape[0])
        loaded.append((item, stem, values, valid, current_frames, current_coords))

    n_samples = len(loaded)
    n_components = len(components)
    Y = np.full((n_samples, max_frames, n_nodes, n_components), np.nan, dtype=dtype)
    valid_mask = np.zeros(Y.shape, dtype=bool)
    sample_node_coords = np.full((n_samples, n_nodes, 2), np.nan, dtype=dtype)
    sample_stems = []
    source_files = []

    for sample_idx, (item, stem, values, valid, current_frames, current_coords) in enumerate(loaded):
        keep = values.shape[0]
        Y[sample_idx, :keep, :, :] = values
        valid_mask[sample_idx, :keep, :, :] = valid
        if current_coords is not None:
            sample_node_coords[sample_idx] = current_coords
        sample_stems.append(stem)
        source_files.append(str(item["path"]))
        if keep > len(frame_values):
            frame_values = current_frames

    if node_coords0 is None:
        node_coords0 = sample_node_coords[0]
    if len(frame_values) < max_frames:
        padded_frames = np.full((max_frames,), np.nan, dtype=float)
        padded_frames[:len(frame_values)] = frame_values
        frame_values = padded_frames

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        Y=Y,
        valid_mask=valid_mask,
        frame_values=np.asarray(frame_values[:max_frames], dtype=float),
        node_labels=np.asarray(node_labels, dtype=int),
        node_coords=np.asarray(node_coords0, dtype=dtype),
        coords0=np.asarray(node_coords0, dtype=dtype),
        sample_node_coords=sample_node_coords,
        sample_ids=np.asarray(sample_ids, dtype=str),
        sample_stems=np.asarray(sample_stems, dtype=str),
        sample_numbers=np.asarray([str(i) for i in sample_numbers], dtype=str),
        elements=np.asarray(elements, dtype=int),
        components=np.asarray(components, dtype=str),
        field_names=np.asarray(field_names or [], dtype=str),
        mode=np.asarray([mode], dtype=str),
        source_field_files=np.asarray(source_files, dtype=str),
    )

    meta = {
        "path": str(out_path),
        "mode": mode,
        "mech_test": mech_test,
        "disorder": dis,
        "shape": list(Y.shape),
        "components": list(components),
        "sample_ids": [str(i) for i in sample_ids],
        "sample_numbers": [str(i) for i in sample_numbers],
        "valid_fraction": float(valid_mask.mean()) if valid_mask.size else np.nan,
        "source_field_files": source_files,
    }
    with open(str(out_path).replace(".npz", ".summary.json"), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_number": sample_numbers,
            "stem": sample_stems,
            "mode": mode,
            "field_file": source_files,
            "processed_file": str(out_path),
        }
    )

def create_fieldNPZ(directory, dat=None, field_dir="", raw_dir="transfer", dis=None, dtype="float32", field_prefix="FIELDu-"):
    """Stack raw per-simulation field outputs into one processed NPZ per mechanical mode."""
    dis = dis or (dat.dis if dat is not None else "disNodes")
    field_family = _field_family_name(field_prefix)
    create_fieldIndexCSV(directory, field_dir=field_dir, raw_dir=raw_dir, field_prefix=field_prefix, dat=dat, dis=dis)

    groups = {"UT": [], "FT": []}
    for path in _field_raw_files(directory, field_dir=field_dir, raw_dir=raw_dir, field_prefix=field_prefix):
        with np.load(path, allow_pickle=True) as npz:
            stem = _npz_first_string(npz, ["sample_stem", "sample_stems"], default=_field_stem_from_path(path, field_prefix))
            mode, mech_test = _field_mode_from_npz(npz, stem)
            if mode not in groups:
                continue
            if dat is not None:
                if mode == "UT" and not dat.UTmechTest:
                    continue
                if mode == "FT" and not dat.FTmechTest:
                    continue
                if not _dis_enabled(dat, dis):
                    continue
            groups[mode].append(
                {
                    "path": path,
                    "stem": stem,
                    "sample_id": _field_sample_id(npz, stem),
                    "mech_test": mech_test,
                }
            )

    out_root = _field_root(directory, field_dir=field_dir)
    UTfield_df = _stack_field_group(
        groups["UT"],
        out_root / f"Ductile-{dis}-FIELDu.npz",
        "UT",
        "Ductile",
        dis,
        dtype=dtype,
    )
    FTfield_df = _stack_field_group(
        groups["FT"],
        out_root / f"Fracture-{dis}-FIELDu.npz",
        "FT",
        "Fracture",
        dis,
        dtype=dtype,
    )

    if UTfield_df is not None:
        UTfield_df.to_csv(out_root / f"Ductile-{dis}-{field_family}-stack-index.csv", index=False)
    if FTfield_df is not None:
        FTfield_df.to_csv(out_root / f"Fracture-{dis}-{field_family}-stack-index.csv", index=False)
    return UTfield_df, FTfield_df

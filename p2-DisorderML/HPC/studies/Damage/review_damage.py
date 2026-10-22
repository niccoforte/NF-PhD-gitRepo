"""Damage-study visual review of named strut NPZs; no ODB/model access."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from resources.damage import require, validate_damage_arrays


FAILED_COLOR = "#d62728"


def binary_fracture(a, entity="strut"):
    """STATUS-derived 0/1 state; unknown is masked independently of severity."""
    return np.where(a[f"{entity}_failure_known"], a[f"{entity}_failed"].astype(float), np.nan)


def fracture_scale():
    return ListedColormap(["#440154", FAILED_COLOR]), BoundaryNorm([-.5, .5, 1.5], 2)


def load(path, task):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k:saved[k] for k in saved.files}
    require(str(arrays["task"]) == task, "Review task mismatch")
    summary = validate_damage_arrays(arrays)
    summary["npz_path"] = str(Path(path).resolve())
    metadata = json.loads(str(arrays["metadata_json"]))
    require(metadata["task"] == task and int(metadata["simulation_id"]) == int(arrays["simulation_id"]),
            "Saved metadata task/simulation identity mismatch")
    return arrays, metadata, summary


def parent_record(a, parent, frame, kind):
    members = np.flatnonzero(a["parent_strut_index"] == parent)
    elements = []
    for index in members:
        points = np.flatnonzero(a["point_element_index"] == index)
        elements.append(dict(label=int(a["element_label"][index]), instance=str(a["element_instance"][index]),
            connectivity=a['element_connectivity'][index].tolist(),
            status=float(a["status"][frame,index]), fracture=float(binary_fracture(a,"element")[frame,index]) if a["element_failure_known"][frame,index] else None, damage=float(a["element_damage"][frame,index]) if np.isfinite(a["element_damage"][frame,index]) else None,
            sdeg=[dict(section=int(a["point_section"][p]), integration=int(a["point_integration"][p]),
                       description=str(a["point_section_description"][p]), value=float(a["sdeg"][frame,p])) for p in points]))
    return dict(task=str(a["task"]), kind=kind, strut_id=int(parent), frame=int(frame),
                time=float(a["frame_time"][frame]), loading=float(a["global_loading"][frame]),
                endpoint_labels=a["endpoint_node_label"][parent].tolist(),
                endpoint_coordinates=a["endpoint_xy0"][parent].tolist(), elements=elements,
                damage=float(a["strut_damage"][frame,parent]) if np.isfinite(a["strut_damage"][frame,parent]) else None,
                fracture=float(binary_fracture(a)[frame,parent]) if a["strut_failure_known"][frame,parent] else None,
                failed_elements=int(a["failed_element_count"][frame,parent]))


def crack_tip(a, metadata):
    """Use an explicit profile landmark, never a saved ML/canonical node index."""
    if str(a["task"]) != "FT":
        return None
    value = metadata.get("nominal_crack_tip")
    if value is None and metadata["profile"] == "periodic_fcc20_body_v1":
        value = [120., 95.]  # checked legacy periodic profile, in initial mm
    require(value is not None, "FT review requires nominal_crack_tip metadata")
    value = np.asarray(value, float)
    require(value.shape == (2,) and np.isfinite(value).all(), "Invalid nominal crack tip")
    return value


def tip_parent(a, metadata):
    tip = crack_tip(a, metadata)
    if tip is None:
        return 0
    starts = a["endpoint_xy0"][:, 0]
    vectors = a["endpoint_xy0"][:, 1] - starts
    along = np.clip(((tip - starts) * vectors).sum(axis=1) / (vectors**2).sum(axis=1), 0, 1)
    distance = np.linalg.norm(starts + along[:, None] * vectors - tip, axis=1)
    return int(np.argmin(distance))


def keyframes(a):
    first = np.flatnonzero(a["strut_failed"].any(axis=1))
    return [0, int(first[0]) if len(first) else len(a["frame_time"])//2, len(a["frame_time"])-1]


def loading_label(a):
    return "UT nominal strain" if str(a["task"]) == "UT" else "FT load-point U2 [mm]"


def select_examples(a, metadata, label):
    failed = a["strut_failed"]
    intact = tip_parent(a, metadata)
    records = [parent_record(a, intact, 0, "intact")]
    point_parent=a["parent_strut_index"][a["point_element_index"]]
    active=observed_partial(a)[:,point_parent]
    candidates=np.argwhere(active & (a["sdeg"]>0))
    if len(candidates):
        frame,point=candidates[0]
        records.append(parent_record(a,int(point_parent[point]),int(frame),"partial"))
    one = np.argwhere(failed & (a["failed_element_count"] == 1) & (a["subdivision_count"][None,:] > 1))
    if len(one):
        frame,parent = one[0]
        records.append(parent_record(a,int(parent),int(frame),"one_failed_element"))
    for record in records:
        record["case"] = label
    return records


def observed_partial(a):
    """Positive reported SDEG in an all-active parent, including a surface at 1."""
    result=np.zeros(a['strut_failed'].shape,bool)
    point_parent=a['parent_strut_index'][a['point_element_index']]
    for parent in range(result.shape[1]):
        raw=a['sdeg'][:,point_parent==parent]
        complete=a['element_sdeg_complete'][:,a['parent_strut_index']==parent].all(axis=1)
        result[:,parent]=(raw>0).any(axis=1) & complete & a['strut_failure_known'][:,parent] & ~a['strut_failed'][:,parent]
    return result


def segments(a, elements, frame=None):
    lookup = {int(n):i for i,n in enumerate(a["node_label"])}
    indices = np.array([[lookup[int(n)] for n in conn] for conn in a["element_connectivity"][elements]])
    xy = a["node_xy0"][indices]
    return xy if frame is None else xy + a["node_displacement"][frame,indices]


def draw_parent(ax, a, parent, frame, annotate=True):
    members = np.flatnonzero(a["parent_strut_index"] == parent)
    xy = segments(a, members, frame)
    cmap, norm = fracture_scale()
    values = binary_fracture(a, "element")[frame,members]
    collection = LineCollection(xy, cmap=cmap, norm=norm, linewidths=4)
    collection.set_array(np.ma.masked_invalid(values))
    ax.add_collection(LineCollection(xy[~np.isfinite(values)],colors='black',linestyles='dashed',linewidths=4))
    ax.add_collection(collection)
    for i,index in enumerate(members):
        ax.scatter(xy[i,:,0],xy[i,:,1],s=14,c="0.3",zorder=3)
        if annotate:
            raw = a["sdeg"][frame,a["point_element_index"] == index]
            ax.annotate(f'e{a["element_label"][index]}',xy[i].mean(axis=0),xytext=(5,-12),textcoords='offset points',fontsize=7)
            text = f'e{a["element_label"][index]}: {"/".join(f"{v:.8g}" for v in raw)}; S={a["status"][frame,index]:g}'
            ax.text(.50,.98-i*.065,text,transform=ax.transAxes,va='top',fontsize=6.5)
    ax.update_datalim(xy.reshape(-1,2))
    ax.autoscale_view()
    xmid,ymid=xy.reshape(-1,2).mean(axis=0)
    span=max(float(np.ptp(xy.reshape(-1,2),axis=0).max()),5)
    ax.set_xlim(xmid-.8*span,xmid+1.4*span)
    ax.set_ylim(ymid-.8*span,ymid+.8*span)
    ax.set_aspect("equal")
    ax.set(xlabel="deformed x [mm]",ylabel="deformed y [mm]")
    ax.grid(alpha=.15)


def maps(cases, output, title, filename="damage-overview.png"):
    fig,axes=plt.subplots(len(cases),3,figsize=(14,4.3*len(cases)),squeeze=False,constrained_layout=True)
    cmap, norm = fracture_scale()
    has_missing = False
    for row,(label,(a,m,_)) in enumerate(cases.items()):
        values = binary_fracture(a)
        for col,(ax,frame) in enumerate(zip(axes[row],keyframes(a))):
            missing = ~np.isfinite(values[frame])
            has_missing |= bool(missing.any())
            ax.add_collection(LineCollection(a["endpoint_xy0"][missing],colors="black",linestyles="dashed",linewidths=1))
            collection=LineCollection(a["endpoint_xy0"],cmap=cmap,norm=norm,linewidths=1)
            collection.set_array(np.ma.masked_invalid(values[frame]))
            ax.add_collection(collection)
            broken=a["strut_failed"][frame]
            if broken.any():
                ax.add_collection(LineCollection(a["endpoint_xy0"][broken],colors=FAILED_COLOR,linewidths=1.8))
            ax.autoscale_view()
            ax.set_aspect("equal")
            value=a["global_loading"][frame]
            loading=f'nominal strain {value:.4f}' if str(a["task"])=="UT" else f'load-point U2 {value:.3f} mm'
            stage = ("Initial", "First recorded fracture" if a["strut_failed"].any() else "Middle recorded frame", "Final")[col]
            ax.set_title(f'{label} | {stage}\nframe {frame}; t={a["frame_time"][frame]:.5f} s\n{loading}; fractured {broken.sum()}',fontsize=9)
            ax.set(xlabel="initial x [mm]",ylabel="initial y [mm]")
    bar = fig.colorbar(collection,ax=axes.ravel().tolist(),label="Binary strut fracture, D_fracture (from STATUS)",shrink=.75)
    bar.set_ticks([0, 1], labels=["0: unfractured", "1: fractured"])
    if has_missing:
        fig.legend(handles=[Line2D([], [], color="black", linestyle="--", linewidth=2,
            label="Unknown fracture state: incomplete STATUS")],loc="outside lower center",fontsize=10)
    scope = " UT body only." if any(str(case[0]["task"]) == "UT" for case in cases.values()) else ""
    fig.suptitle(f"{title}: binary fracture on initial geometry\nD_fracture=1 if any member STATUS=0; otherwise 0 when all active.{scope}",fontsize=13)
    fig.savefig(output/filename,dpi=160)
    plt.close(fig)


def examples_figure(cases, records, output):
    fig,axes=plt.subplots(len(cases),3,figsize=(19,4.8*len(cases)),squeeze=False,constrained_layout=True)
    for row,label in enumerate(cases):
        a=cases[label][0]
        chosen=[r for r in records if r["case"]==label]
        for col,kind in enumerate(("intact","partial","one_failed_element")):
            ax=axes[row,col]
            found=[r for r in chosen if r["kind"]==kind]
            if not found:
                ax.text(.5,.5,f"{label}: no {kind.replace('_',' ')} example observed",ha="center",transform=ax.transAxes)
                ax.set_axis_off()
                continue
            r=found[0]
            draw_parent(ax,a,r["strut_id"],r["frame"])
            arithmetic = f"{r['fracture']:g}" if r['fracture'] is not None else "unknown (incomplete STATUS)"
            endpoints=" -> ".join(f'n{n} ({xy[0]:.3f}, {xy[1]:.3f})' for n,xy in zip(r["endpoint_labels"],r["endpoint_coordinates"]))
            state = "active with surface damage" if kind == "partial" else kind.replace("_"," ")
            ax.set_title(f'{label}: {state}; strut {r["strut_id"]}, frame {r["frame"]}\nD_fracture={arithmetic}; failed members={r["failed_elements"]}\nInitial endpoints [mm]: {endpoints}',fontsize=9)
    fig.suptitle("Binary fracture colours: purple=0, red=1; S=whole-element STATUS (0=failed)\nRaw bottom/top SDEG values are diagnostic annotations, not the plotted fracture variable. True-scale deformation.",fontsize=13)
    fig.savefig(output/"damage-examples.png",dpi=150)
    plt.close(fig)


def histories(cases, records, output):
    fig,axes=plt.subplots(len(cases),2,figsize=(13,3.6*len(cases)),squeeze=False,constrained_layout=True)
    for row,(case_label,(a,m,_)) in enumerate(cases.items()):
        x=a["global_loading"]
        label=loading_label(a)
        seen=set()
        for r in sorted(records,key=lambda r:r['kind']!='partial'):
            if r["case"]==case_label:
                parent=r['strut_id']
                if parent in seen:
                    continue
                seen.add(parent)
                axes[row,0].plot(x,binary_fracture(a)[:,parent],"o-",drawstyle="steps-post",markersize=3,label=f'parent {parent}')
        axes[row,0].set(xlabel=label,ylabel="D_fracture (STATUS only)",yticks=[0,1],ylim=(-.03,1.06),title=f"{case_label}: selected parent histories ({len(x)} recorded frames)")
        axes[row,0].legend(fontsize=8)
        axes[row,1].step(x,a["strut_failed"].sum(axis=1),where="post",label="failed parents")
        partial=observed_partial(a).sum(axis=1)
        axes[row,1].plot(x,partial,"o-",markersize=3,label="active with surface damage (SDEG diagnostic)")
        axes[row,1].set(xlabel=label,ylabel="Number of struts",title=f'{case_label}: fractures and separate surface-damage counts')
        axes[row,1].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=.2)
    fig.suptitle('Markers are recorded frames; between-frame softening/onset is unresolved',fontsize=12)
    fig.savefig(output/"damage-histories.png",dpi=160)
    plt.close(fig)


def tip_figure(cases, output):
    ft_cases = [(label, case) for label, case in cases.items() if str(case[0]["task"]) == "FT"]
    if not ft_cases:
        return
    fig, axes = plt.subplots(len(ft_cases), 3, figsize=(14, 4.2*len(ft_cases)), squeeze=False, constrained_layout=True)
    cmap, norm = fracture_scale()
    for row, (label, (a, metadata, _)) in enumerate(ft_cases):
        tip = crack_tip(a, metadata)
        for ax, frame in zip(axes[row], keyframes(a)):
            values = binary_fracture(a)[frame]
            lines = LineCollection(a["endpoint_xy0"], cmap=cmap, norm=norm, linewidths=1.5)
            lines.set_array(np.ma.masked_invalid(values))
            ax.add_collection(lines)
            ax.add_collection(LineCollection(a["endpoint_xy0"][~np.isfinite(values)], colors="black", linestyles="dashed", linewidths=1.5))
            broken = a["strut_failed"][frame]
            ax.add_collection(LineCollection(a["endpoint_xy0"][broken], colors=FAILED_COLOR, linewidths=2.5))
            ax.scatter(*tip, marker="x", color="#e6a800", s=55, zorder=4, label="nominal initial tip")
            ax.set(xlim=(tip[0]-25, tip[0]+35), ylim=(tip[1]-30, tip[1]+30), aspect="equal",
                   xlabel="initial x [mm]", ylabel="initial y [mm]",
                   title=f'{label}: frame {frame}; U2={a["global_loading"][frame]:.3f} mm')
            ax.legend(fontsize=8, loc="lower left")
    bar = fig.colorbar(lines, ax=axes.ravel().tolist(), shrink=.75, label="D_fracture from STATUS")
    bar.set_ticks([0, 1], labels=["0: unfractured", "1: fractured"])
    fig.suptitle("FT crack-tip region on initial geometry; nominal tip is a reference point, not an assumed mesh node", fontsize=12)
    fig.savefig(output/"damage-tip-history.png", dpi=160)
    plt.close(fig)


def compare_sources(cases):
    """Check selected identity/response arrays; filenames alone establish neither."""
    comparisons = []
    mesh_keys = ("endpoint_xy0", "canonical_endpoint", "element_label", "element_connectivity",
                 "parent_strut_index", "node_label", "node_xy0")
    response_keys = ("frame_time", "status", "strut_failed", "strut_failure_known", "sdeg", "ductcrt",
                     "node_displacement", "global_loading", "load_displacement", "reaction_force",
                     "history_time", "history_value")
    for i, (left, (a, ma, _)) in enumerate(cases.items()):
        for right, (b, mb, _) in list(cases.items())[i+1:]:
            if ma["source_sha256"]["inp"] == mb["source_sha256"]["inp"]:
                mesh = all(np.array_equal(a[k], b[k], equal_nan=True) for k in mesh_keys)
                matches = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in response_keys}
                differing = [k for k in sorted(a.keys() | b.keys()) if k != "metadata_json" and
                             (k not in a or k not in b or not np.array_equal(
                                 a[k], b[k], equal_nan=np.issubdtype(a[k].dtype, np.inexact)))]
                comparisons.append(dict(kind="identical_input", left=left, right=right, identical_inp_sha256=True,
                                        identical_selected_mesh=mesh, response_arrays_equal=matches,
                                        different_nonmetadata_arrays=differing,
                                        all_nonmetadata_arrays_equal=not differing))
            if str(a["task"]) != str(b["task"]) and ma["simulation_id"] == mb["simulation_id"]:
                joints_a = {int(node): xy for pair, points in zip(a["canonical_endpoint"], a["endpoint_xy0"])
                            for node, xy in zip(pair, points)}
                joints_b = {int(node): xy for pair, points in zip(b["canonical_endpoint"], b["endpoint_xy0"])
                            for node, xy in zip(pair, points)}
                shared = sorted(joints_a.keys() & joints_b.keys())
                equal = all(np.array_equal(joints_a[node], joints_b[node]) for node in shared)
                edges_a, edges_b = map(lambda values: {tuple(pair) for pair in values},
                                     (a["canonical_endpoint"], b["canonical_endpoint"]))
                comparisons.append(dict(kind="paired_shared_geometry", left=left, right=right,
                                        shared_original_joints=len(shared), shared_joint_coordinates_equal=equal,
                                        shared_parent_endpoint_pairs=len(edges_a & edges_b),
                                        union_parent_endpoint_pairs=len(edges_a | edges_b)))
    return comparisons


def selection_context(metadata):
    stem = metadata["source_stem"]
    if stem.endswith("maxD"):
        return "Researcher-selected highest UT ductility (D)."
    if stem.endswith("maxK"):
        return "Researcher-selected initial FT fracture toughness (K)."
    if stem.endswith("maxB"):
        return "Researcher-selected UT/FT property-Pareto pair (B)."
    return "Periodic reference."


def evidence_link(path, output):
    """Keep local report links usable when the input NPZ is stored elsewhere."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), output.resolve())).as_posix()
    except ValueError:  # a Windows archive and report can be on different drives
        return Path(path).resolve().as_posix()


def write_report(cases, records, output, title):
    result_rows, observation_rows, source_rows, details, evidence = [], [], [], [], []
    for label, (a, metadata, summary) in cases.items():
        partial = observed_partial(a)
        counts = dict(zip(*np.unique(a["subdivision_count"], return_counts=True)))
        first = np.flatnonzero(a["strut_failed"].any(axis=1))
        fates = []
        for parent in np.flatnonzero(partial.any(axis=0)):
            observed = int(np.flatnonzero(partial[:, parent])[0])
            later = np.flatnonzero(a["strut_failed"][observed+1:, parent])
            failure = int(observed+1+later[0]) if len(later) else -1
            fates.append(dict(strut_id=int(parent), first_active_surface_damage_frame=observed,
                              later_fracture_frame=failure,
                              final_failure_known=bool(a["strut_failure_known"][-1, parent]),
                              final_failed=bool(a["strut_failed"][-1, parent])))
        with (output/f"{label}-surface-damage-fate.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["strut_id", "first_active_surface_damage_frame",
                "later_fracture_frame", "final_failure_known", "final_failed"])
            writer.writeheader()
            writer.writerows(fates)
        verification_path = Path(summary["npz_path"]).with_suffix(".verification.json")
        verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
        if verification is not None:
            require(verification["all_frame_parent_failures_verified"] and verification["read_only"],
                    f"Independent ODB verification is incomplete for {label}")
        summary.update(task=str(a["task"]), simulation_id=metadata["simulation_id"],
                       profile=metadata["profile"], source_stem=metadata["source_stem"],
                       first_failure_frame=int(first[0]) if len(first) else -1,
                       first_failure_time=float(a["frame_time"][first[0]]) if len(first) else None,
                       first_failure_loading=float(a["global_loading"][first[0]]) if len(first) else None,
                       final_loading=float(a["global_loading"][-1]), loading_units=metadata["loading_units"],
                       loading_definition=metadata["loading_definition"],
                       final_partial_struts=int(partial[-1].sum()), partial_rule=str(a["partial_rule"]),
                       known_fracture_labels=int(a["strut_failure_known"].sum()),
                       unknown_fracture_labels=int((~a["strut_failure_known"]).sum()),
                       binary_fracture_source="strut_failed with strut_failure_known; whole-element STATUS",
                       ever_active_surface_damage=len(fates),
                       active_surface_damage_later_fractured=sum(r["later_fracture_frame"] >= 0 for r in fates),
                       active_surface_damage_final_active=sum(r["final_failure_known"] and not r["final_failed"] for r in fates),
                       missing_status=int((~a["status_valid"]).sum()), missing_sdeg=int((~a["sdeg_valid"]).sum()),
                       subdivision_histogram={str(k): int(v) for k,v in counts.items()},
                       excluded_elements=int(metadata["mapping"]["excluded_elements"]),
                       source_sha256=metadata["source_sha256"],
                       source_unchanged_sha256=metadata["source_unchanged_sha256"],
                       status_reactivations=metadata["status_reactivations"], live_odb_verification=verification)
        fracture = summary["final_failed_struts"]
        first_loading = f"{summary['first_failure_loading']:.6g}" if len(first) else "not observed"
        result_rows.append(f"| {label} | {summary['simulation_id']} | {summary['struts']:,} | {summary['elements']:,} | **{fracture}** | {summary['first_failure_frame']} | {first_loading} | {summary['final_loading']:.6g} |")
        observation_rows.append(f"| {label} | {summary['frames']} | {summary['excluded_elements']} | {summary['missing_status']} | {summary['unknown_fracture_labels']} | {summary['subdivision_histogram']} |")
        source_rows.append(f"| {label} | `{metadata['source_stem']}` | {selection_context(metadata)} |")
        evidence.append(f"- **{label}**: [NPZ]({evidence_link(summary['npz_path'], output)}), "
                        f"[extraction metadata]({evidence_link(Path(summary['npz_path']).with_suffix('.json'), output)}), "
                        f"[independent ODB verification]({evidence_link(verification_path, output)}), "
                        f"[frame table]({label}-frame-summary.csv), "
                        f"[surface-damage fates]({label}-surface-damage-fate.csv).")
        details.append(f"- **{label}**: {len(fates)} struts are observed active with surface damage; "
                       f"{summary['active_surface_damage_later_fractured']} later fracture and "
                       f"{summary['active_surface_damage_final_active']} remain known active at the final frame.")
        with (output/f"{label}-frame-summary.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "time_s", "global_loading", "reaction_force_N", "failed_struts",
                             "active_surface_damage_struts", "fracture_known_struts", "severity_valid_struts"])
            for j,time in enumerate(a["frame_time"]):
                writer.writerow([j, time, a["global_loading"][j], a["reaction_force"][j],
                                 a["strut_failed"][j].sum(), partial[j].sum(),
                                 a["strut_failure_known"][j].sum(), a["strut_valid"][j].sum()])
    comparisons = compare_sources(cases)
    repeated = []
    for comparison in comparisons:
        if comparison["kind"] == "paired_shared_geometry":
            repeated.append(f"**{comparison['left']} / {comparison['right']}** share "
                            f"{comparison['shared_original_joints']} checked original joints, whose coordinates "
                            f"{'match exactly' if comparison['shared_joint_coordinates_equal'] else 'differ'}, "
                            f"and {comparison['shared_parent_endpoint_pairs']} parent endpoint pairs. "
                            "FT removes initial crack-region geometry; it is not labelled as failed at frame 0.")
            continue
        same_response = all(comparison["response_arrays_equal"].values())
        complete = (" All saved arrays outside source/provenance metadata match exactly."
                    if comparison["all_nonmetadata_arrays_equal"] else "")
        repeated.append(f"**{comparison['left']} and {comparison['right']} have byte-identical INPs.** "
                        f"Selected mesh arrays are {'identical' if comparison['identical_selected_mesh'] else 'different'}; "
                        f"the inspected saved response arrays are {'also identical' if same_response else 'not all identical'}.{complete} "
                        "They must not be counted as independent geometries or independent evidence of generalisation. "
                        "Their source ODB fingerprints remain recorded separately.")
    if not repeated:
        repeated.append("No byte-identical INP pair occurs among these explicit selections.")
    verifications = [case[2]["live_odb_verification"] for case in cases.values()]
    if all(v is not None for v in verifications):
        status_count = sum(v["raw_status_values_verified"] for v in verifications)
        sdeg_count = sum(v["raw_sdeg_values_spotchecked"] for v in verifications)
        check_text = (f"Independent read-only ODB checks matched **{status_count:,} STATUS values**, "
                      f"all-frame parent failures, and **{sdeg_count:,} section SDEG spot checks**.")
    else:
        check_text = "Independent live-ODB verification is not yet available for every selected NPZ; see per-case evidence."
    fractured = "; ".join(f"**{label}: {case[2]['final_failed_struts']}**" for label,case in cases.items())
    unknown = sum(case[2]["unknown_fracture_labels"] for case in cases.values())
    known = sum(case[2]["known_fracture_labels"] for case in cases.values())
    task_views = ("For larger readable panels: [UT full-body views](damage-overview-UT.png), "
                  "[FT full-body views](damage-overview-FT.png)." if len(cases) > 2 else "")
    result_text, observation_text, source_text = map("\n".join, (result_rows, observation_rows, source_rows))
    duplicate_text, details_text, evidence_text = map("\n\n".join, (repeated, details, evidence))
    report = f"""# {title}: strut fracture results

**STATUS alone defines binary fracture.** Final fractured struts: {fractured}.
There are **{known:,} known** parent/frame labels and **{unknown:,} unknown** labels.
This report reviews explicit source files; it does not fit a model or rank new property optima.

![Initial, first-fracture and final full-body binary fracture maps](damage-overview.png)

{task_views}

| Colour | D_fracture | Meaning |
| --- | ---: | --- |
| Purple | 0 | All constituent elements known active (STATUS=1); local surface damage may exist. |
| Red | 1 | At least one constituent element has failed (STATUS=0). |
| Black dashed, only if needed | Unknown | No verified failed member and incomplete STATUS; never filled with 0. |

There is no grey surface-damage category. Missing initial FT struts are absent geometry,
not fractured struts. UT maps and counts include the main body only, excluding grips.

## Selected specimens and main results

| Case | Actual source stem | Selection context supplied by researcher |
| --- | --- | --- |
{source_text}

D denotes highest UT ductility, K initial FT fracture toughness, and B the selected
same-disorder UT/FT property-Pareto pair. These are source-selection meanings, not
properties newly calculated by this damage audit. Simulation IDs and profile identity
come from each saved product's checked source metadata; display labels are not IDs.

| Case | Simulation ID | Native struts | Body elements | Final fractured | First fracture frame | First fracture loading | Final loading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{result_text}

UT loading is nominal strain (mean top-grip U2 / initial body height); FT loading is
load-point U2 in mm, not material strain. Each case uses its own actual recorded time
and loading values. The first-fracture column is an observed frame index; precise
onset within the preceding frame interval is unresolved.

{duplicate_text}

## FT crack-tip region

![Binary fracture near the nominal initial FT crack tip](damage-tip-history.png)

The gold marker is the nominal initial crack-tip landmark supplied by the checked
profile. Actual disordered strut coordinates and connectivity are plotted. The marker
is not assumed to be a mesh node; a removed initial crack-region connection is not a
STATUS failure. Red colours show fractured parent struts, not an interpolated continuum crack.

## Fracture over loading

![Selected binary parent histories and population fracture counts](damage-histories.png)

Markers are recorded frames. Step displays carry a known sample value to the next
sample for readability; they do not determine the physical failure instant. SDEG
surface-damage counts are a separately labelled diagnostic, never the fracture target.

## Constituent-element examples

![Intact, active surface-damaged and one-failed-member parent examples per case](damage-examples.png)

Each row is a source case. The examples preserve actual original endpoint/element
labels and per-member STATUS. Parent D_fracture becomes 1 when one member has STATUS=0,
regardless of subdivision count. Purple active members can have positive SDEG without
whole-element failure. Endpoints are initial coordinates; example elements are drawn
at true-scale recorded deformation. Raw section SDEG annotations are diagnostics.
An absent active surface-damage example means none is observed at the recorded frames.

{details_text}

Those final active states do not predict eventual fracture under further loading.
Bottom/top SDEG refers to opposite local beam-section surfaces, not element endpoints.
The interior section state is not inferred from the reported outer surfaces. No
near-one SDEG threshold or aggregation choice is needed for binary fracture.

## Observation coverage and scope

| Case | Recorded frames | Excluded grip elements | Missing STATUS records | Unknown fracture labels | Elements per strut: count |
| --- | ---: | ---: | ---: | ---: | --- |
{observation_text}

All checks use native struts and instance-scoped original FE identities. Audit
canonical endpoint indices are not saved ML node order. No strut target is broadcast
onto nodes or passed into current DATA/MODEL loaders. Frame grids are never joined by
ordinal position across files. Separate per-case masks and provenance remain available.

## Evidence and use

{check_text}

{evidence_text}

Use saved **strut_failed** with **strut_failure_known** (and element equivalents):
1 for verified failure, 0 for known all-active, unknown otherwise. Optional scalar
`strut_damage` values are a different quantity and may be NaN even when binary
fracture is fully known. Raw SDEG/DUCTCRT is diagnostic; STATUS establishes failure.

The [summary/provenance](results-summary.json), [source comparisons](source-comparisons.json)
and [raw constituent examples](examples.json) support this report. Source fingerprints,
original membership, raw STATUS/section values, and actual loading/time are retained
in the NPZs and adjacent extraction JSON. Original artifacts are read-only; generated
research products remain ignored by Git. No simulation, model architecture, training,
or archive-wide discovery is part of this report.

Maintained implementation: p1 `A2_DamageOUTpostProcess.py`, shared `resources/damage.py`,
p2 `HPC/studies/Damage/review_damage.py`. The [quick start](../../../p1-DisorderLatticeProperties/SIMscripts/DAMAGE.md)
contains execution commands; the full array specification is an optional reference.
"""
    (output/"README.md").write_text(report, encoding="utf-8")
    (output/"examples.json").write_text(json.dumps(records, indent=2)+"\n", encoding="utf-8")
    (output/"results-summary.json").write_text(json.dumps({label:case[2] for label,case in cases.items()}, indent=2)+"\n", encoding="utf-8")
    (output/"source-comparisons.json").write_text(json.dumps(comparisons, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", help="Periodic-compatible explicit UT NPZ, paired with --ft")
    parser.add_argument("--ft", help="Periodic-compatible explicit FT NPZ, paired with --ut")
    parser.add_argument("--case", action="append", nargs=3, metavar=("LABEL", "TASK", "NPZ"),
                        help="Explicit case, repeat for a comparison; TASK must be UT or FT")
    parser.add_argument("--title", help="Human-readable comparison title")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    require(not (args.case and (args.ut or args.ft)), "Use --case or the paired --ut/--ft invocation")
    if args.case:
        selected = args.case
    else:
        require(args.ut and args.ft, "Supply repeated --case LABEL TASK NPZ, or both --ut and --ft")
        selected = [("UT", "UT", args.ut), ("FT", "FT", args.ft)]
    require(len({label for label,_,_ in selected}) == len(selected), "Case labels must be unique")
    require(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) for label,_,_ in selected),
            "Case labels must be safe alphanumeric file stems with optional '-' or '_'")
    require(all(task in {"UT", "FT"} for _,task,_ in selected), "Case tasks must be UT or FT")
    cases = {label:load(path,task) for label,task,path in selected}
    periodic = all(case[1]["profile"] == "periodic_fcc20_body_v1" for case in cases.values())
    title = args.title or ("Periodic UT/FT" if periodic else "Disordered FCC")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = [record for label,(a,metadata,_) in cases.items()
               for record in select_examples(a, metadata, label)]
    maps(cases, output, title)
    if len(cases) > 2:
        for task in ("UT", "FT"):
            subset = {label:case for label,case in cases.items() if str(case[0]["task"]) == task}
            if subset:
                maps(subset, output, title, f"damage-overview-{task}.png")
    examples_figure(cases, records, output)
    histories(cases, records, output)
    tip_figure(cases, output)
    write_report(cases, records, output, title)
    print(json.dumps({label:case[2] for label,case in cases.items()}, indent=2))


if __name__ == "__main__":
    main()

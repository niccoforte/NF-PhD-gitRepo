"""Readable review of two explicit strut-damage NPZs; no ODB/model access."""
import argparse
import csv
import json
from pathlib import Path

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
    return arrays, json.loads(str(arrays["metadata_json"])), summary


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


def select_examples(a):
    failed = a["strut_failed"]
    tip = np.flatnonzero((a["canonical_endpoint"] == [381,422]).all(axis=1))
    intact = int(tip[0]) if len(tip) else 0
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
            text = f'e{a["element_label"][index]}: {"/".join(f"{v:.6f}" for v in raw)}; S={a["status"][frame,index]:g}'
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


def maps(cases, output):
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    cmap, norm = fracture_scale()
    has_missing = False
    for row,(task,(a,m,_)) in enumerate(cases.items()):
        values = binary_fracture(a)
        first=np.flatnonzero(a["strut_failed"].any(axis=1))
        selected=[0,int(first[0]) if len(first) else len(a["frame_time"])//2,len(a["frame_time"])-1]
        for ax,frame in zip(axes[row],selected):
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
            loading=f'nominal strain {value:.4f}' if task=="UT" else f'load-point U2 {value:.3f} mm'
            ax.set(title=f'{task}: frame {frame}, t={a["frame_time"][frame]:.5f} s\n{loading}; fractured {broken.sum()}',xlabel="initial x [mm]",ylabel="initial y [mm]")
    bar = fig.colorbar(collection,ax=axes.ravel().tolist(),label="Binary strut fracture, D_fracture (from STATUS)",shrink=.75)
    bar.set_ticks([0, 1], labels=["0: unfractured", "1: fractured"])
    if has_missing:
        fig.legend(handles=[Line2D([], [], color="black", linestyle="--", linewidth=2,
            label="Unknown fracture state: incomplete STATUS")],loc="outside lower center",fontsize=10)
    fig.suptitle("Periodic UT/FT: binary fracture on initial geometry\nD_fracture=1 if any member STATUS=0; otherwise 0 when all active. UT body only.",fontsize=13)
    fig.savefig(output/"damage-overview.png",dpi=160)
    plt.close(fig)


def examples_figure(cases, records, output):
    fig,axes=plt.subplots(2,3,figsize=(19,10),constrained_layout=True)
    for row,task in enumerate(cases):
        a=cases[task][0]
        chosen=[r for r in records if r["task"]==task]
        for col,kind in enumerate(("intact","partial","one_failed_element")):
            ax=axes[row,col]
            found=[r for r in chosen if r["kind"]==kind]
            if not found:
                ax.text(.5,.5,f"{task}: no {kind} example observed",ha="center",transform=ax.transAxes)
                ax.set_axis_off()
                continue
            r=found[0]
            draw_parent(ax,a,r["strut_id"],r["frame"])
            arithmetic = f"{r['fracture']:g}" if r['fracture'] is not None else "unknown (incomplete STATUS)"
            endpoints=" -> ".join(f'n{n} {tuple(xy)}' for n,xy in zip(r["endpoint_labels"],r["endpoint_coordinates"]))
            state = "active with surface damage" if kind == "partial" else kind.replace("_"," ")
            ax.set_title(f'{task} {state}: strut {r["strut_id"]}, frame {r["frame"]}\nD_fracture={arithmetic}; failed members={r["failed_elements"]}\nInitial endpoints {endpoints}',fontsize=9)
    fig.suptitle("Binary fracture colours: purple=0, red=1; S=whole-element STATUS (0=failed)\nRaw bottom/top SDEG values are diagnostic annotations, not the plotted fracture variable. True-scale deformation.",fontsize=13)
    fig.savefig(output/"damage-examples.png",dpi=150)
    plt.close(fig)


def histories(cases, records, output):
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,(task,(a,m,_)) in enumerate(cases.items()):
        x=a["global_loading"]
        label="UT nominal strain: mean top-grip U2 / body H" if task=="UT" else "FT load-point U2 [mm]"
        seen=set()
        for r in sorted(records,key=lambda r:r['kind']!='partial'):
            if r["task"]==task:
                parent=r['strut_id']
                if parent in seen:
                    continue
                seen.add(parent)
                axes[row,0].plot(x,binary_fracture(a)[:,parent],"o-",drawstyle="steps-post",markersize=3,label=f'parent {parent}')
        axes[row,0].set(xlabel=label,ylabel="D_fracture (STATUS only)",yticks=[0,1],ylim=(-.03,1.06),title=f"{task}: selected parent fracture histories ({len(x)} recorded frames)")
        axes[row,0].legend(fontsize=8)
        axes[row,1].step(x,a["strut_failed"].sum(axis=1),where="post",label="failed parents")
        partial=observed_partial(a).sum(axis=1)
        axes[row,1].plot(x,partial,"o-",markersize=3,label="active with surface damage (SDEG diagnostic)")
        axes[row,1].set(xlabel=label,ylabel="Number of struts",title=f'{task}: fracture counts and separate surface-damage diagnostic')
        axes[row,1].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=.2)
    fig.suptitle('Markers are recorded frames; between-frame softening/onset is unresolved',fontsize=12)
    fig.savefig(output/"damage-histories.png",dpi=160)
    plt.close(fig)


def tip_figure(case, output):
    a,m,_=case
    parent=int(np.flatnonzero((a["canonical_endpoint"]==[381,422]).all(axis=1))[0])
    fig,axes=plt.subplots(1,4,figsize=(18,4.8),constrained_layout=True)
    for ax,frame in zip(axes[:3],(0,10,20)):
        draw_parent(ax,a,parent,frame)
        value=binary_fracture(a)[frame,parent]
        shown=f'{value:.4g}' if np.isfinite(value) else 'unknown'
        ax.set_title(f'FT tip strut {parent}: frame {frame}\nU2={a["global_loading"][frame]:.3f} mm; D_fracture={shown}',fontsize=10)
    axes[3].plot(a["global_loading"],binary_fracture(a)[:,parent],"o-",drawstyle="steps-post",label="parent D_fracture (STATUS)")
    for index in np.flatnonzero(a["parent_strut_index"]==parent):
        for point in np.flatnonzero(a['point_element_index']==index):
            axes[3].plot(a['global_loading'],a['sdeg'][:,point],'.--',markersize=2,linewidth=.8,label=f'e{a["element_label"][index]} s{a["point_section"][point]}')
    axes[3].set(xlabel="FT load-point U2 [mm]",ylabel="D_fracture / raw section SDEG",ylim=(-.02,1.05),title="Binary fracture and separate surface diagnostics")
    axes[3].legend(fontsize=7)
    fig.suptitle("Crack-adjacent parent: initial endpoints (120,90) and (120,100); deformed geometry at true scale",fontsize=12)
    fig.savefig(output/"damage-tip-history.png",dpi=160)
    plt.close(fig)


def write_report(cases,records,output):
    rows=[]
    for task,(a,m,summary) in cases.items():
        partial=observed_partial(a)
        counts=dict(zip(*np.unique(a["subdivision_count"],return_counts=True)))
        first=np.flatnonzero(a["strut_failed"].any(axis=1))
        fates=[]
        for parent in np.flatnonzero(partial.any(axis=0)):
            observed=int(np.flatnonzero(partial[:,parent])[0])
            later=np.flatnonzero(a["strut_failed"][observed+1:,parent])
            failure_frame=int(observed+1+later[0]) if len(later) else -1
            fates.append(dict(strut_id=int(parent),first_active_surface_damage_frame=observed,
                              later_fracture_frame=failure_frame,
                              final_failure_known=bool(a["strut_failure_known"][-1,parent]),
                              final_failed=bool(a["strut_failed"][-1,parent])))
        with (output/f'{task}-surface-damage-fate.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['strut_id','first_active_surface_damage_frame',
                'later_fracture_frame','final_failure_known','final_failed'])
            writer.writeheader()
            writer.writerows(fates)
        summary.update(first_failure_frame=int(first[0]) if len(first) else -1,
                       first_failure_time=float(a["frame_time"][first[0]]) if len(first) else None,
                       first_failure_loading=float(a["global_loading"][first[0]]) if len(first) else None,
                       final_partial_struts=int(partial[-1].sum()), partial_rule=str(a['partial_rule']),
                       known_fracture_labels=int(a['strut_failure_known'].sum()),
                       binary_fracture_source='strut_failed with strut_failure_known; whole-element STATUS',
                       ever_active_surface_damage=len(fates),
                       active_surface_damage_later_fractured=sum(r['later_fracture_frame'] >= 0 for r in fates),
                       active_surface_damage_final_active=sum(r['final_failure_known'] and not r['final_failed'] for r in fates),
                       missing_status=int((~a["status_valid"]).sum()),missing_sdeg=int((~a["sdeg_valid"]).sum()),
                       subdivision_histogram={str(k):int(v) for k,v in counts.items()},
                       source_sha256=m["source_sha256"],source_unchanged_sha256=m["source_unchanged_sha256"],
                       status_reactivations=m["status_reactivations"])
        rows.append(f'| {task} | {summary["struts"]:,} | {summary["elements"]:,} | {summary["final_failed_struts"]} | {summary["final_partial_struts"]} | {summary["first_failure_frame"]} |')
        with (output/f'{task}-frame-summary.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['frame','time_s','global_loading','reaction_force_N','failed_struts',
                             'active_surface_damage_struts','fracture_known_struts','severity_valid_struts'])
            for j,time in enumerate(a["frame_time"]):
                writer.writerow([j,time,a["global_loading"][j],a["reaction_force"][j],a["strut_failed"][j].sum(),
                                 partial[j].sum(),a["strut_failure_known"][j].sum(),a["strut_valid"][j].sum()])
    ut, ft = cases['UT'][2], cases['FT'][2]
    rows_text="\n".join(rows)
    example=next((r for r in records if r['task']=='UT' and r['kind']=='partial'),None)
    if example:
        damaged=next(e for e in example['elements'] if any(p['value']>0 for p in e['sdeg']))
        raw=', '.join(f"section {p['section']}: {p['value']:.10g}" for p in damaged['sdeg'])
        a=cases['UT'][0]
        later=np.flatnonzero(a['strut_failed'][example['frame']+1:,example['strut_id']])
        outcome=(f"The parent later fractures at frame {example['frame']+1+int(later[0])}." if len(later)
                 else f"The parent does not fracture during the recorded run, which ends at frame {len(a['frame_time'])-1}.")
        example_text=(f"UT **strut {example['strut_id']}, frame {example['frame']}**: element **{damaged['label']}** "
            f"reports {raw}, but STATUS=1. All parent members are active, so **D_fracture=0**. {outcome}")
    else:
        example_text='No UT active surface-damage example is present on the recorded frames.'
    report=f"""# Periodic strut fracture results

**Binary fracture is the target: use STATUS, not SDEG.** Final fractured struts:
UT **{ut['final_failed_struts']}**, FT **{ft['final_failed_struts']}**. Active struts with surface damage
remain **D_fracture=0**, because no member has failed.

![Binary fracture on full UT and FT bodies](damage-overview.png)

| Colour | D_fracture | Meaning |
| --- | ---: | --- |
| Purple | 0 | Every constituent element is known active (STATUS=1). Surface damage may already exist. |
| Red | 1 | At least one constituent element is failed (STATUS=0). |
| Black dashed, only if needed | Unknown | No verified failed member and incomplete STATUS. Never fill this with 0. |

There is **no grey active-damage category** on these fracture plots. Every strut
in the two selected files has a known fracture state. Missing initial FT struts
are absent geometry, not fractured struts.

## Results at the recorded frames

| Task | Struts | Body elements | Final fractured | Final active with surface damage | First fracture frame |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows_text}

Recorded frames including frame 0: UT **{ut['frames']}**, FT **{ft['frames']}**.
UT excludes **830 grip elements**. Subdivision counts vary: UT 4/5, FT 1/2/4/5.
Expected STATUS/SDEG records missing: UT {ut['missing_status']}/{ut['missing_sdeg']};
FT {ft['missing_status']}/{ft['missing_sdeg']}. No STATUS reactivation was observed.

## Does surface damage eventually become fracture?

Among UT struts observed **active with positive surface SDEG**, **{ut['ever_active_surface_damage']}** occur
at some recorded frame. Of these, **{ut['active_surface_damage_later_fractured']} later fracture** and
**{ut['active_surface_damage_final_active']} remain active at the final frame**. This does not predict what happens
under further loading beyond the simulated run. FT has **{ft['ever_active_surface_damage']}** such all-active parents
on its recorded frames; it need not resolve a precursor before fracture.

{example_text}

![Binary element and parent fracture with raw section values alongside](damage-examples.png)

The example JSON retains both `fracture` (the binary label) and `damage`
(the previously saved optional severity, which may be null). They are different
quantities. The plot colours and D_fracture labels use **fracture only**.
Per-parent follow-up: [UT surface-damage fates](UT-surface-damage-fate.csv),
[FT surface-damage fates](FT-surface-damage-fate.csv).

## Why can bottom SDEG be almost 1 while the element is active?

Bottom and top are opposite **surfaces of the local beam cross-section**, not the
two ends of the element or global specimen top/bottom. Bending can give those
surfaces different strains and damage histories. A fully degraded surface point
does not mean the entire cross-section has failed.

These INPs use B21 beams with rectangular sections, default section integration,
and no explicit section-control deletion override. The documented default is
five section points across the planar rectangular beam; only outer points **1
and 5** are saved in these ODBs. The general Explicit deletion condition requires
all section points at an integration location to fail. The interior values are
not saved, so even two degraded outer surfaces do not establish interior failure.
See [beam sections](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-beamcrosssectlib.htm)
and [material failure and deletion](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-materialdata.htm).
These accessible references are for 2025; the actual exports used Abaqus 2024.

The observed stored maximum 0.9999899864 is not used as a failure threshold.
**STATUS=0 is the whole-element flag.** SDEG is retained as optional diagnostic
context; no partial aggregation choice is needed for binary fracture.

## Fracture over loading

![Binary parent fracture histories and separate diagnostic counts](damage-histories.png)

UT horizontal axis: actual field-frame mean SET-TOP U2 / initial body height
**190 mm** (nominal strain). FT: actual LOAD reference-point **U2 in mm**, not
material strain. Binary histories use recorded STATUS; precise onset between
frames is unresolved. Counts of active surface damage are labelled separately.

![Crack-adjacent binary fracture and raw section diagnostic histories](damage-tip-history.png)

## Use and provenance

Use existing NPZ arrays **strut_failed** and **strut_failure_known**. D_fracture
is 1 for a verified failed parent, 0 for a known all-active parent, and unknown
otherwise. `element_failed`/`element_failure_known` provide the element equivalent.
The optional saved `strut_damage` severity arrays are not this binary target;
their original failure-only NaNs do not make fracture unknown. No NPZ or schema
migration or new ODB extraction was needed.

Tracked code: p1 `A2_DamageOUTpostProcess.py`, shared `resources/damage.py`, p2
`code/review_damage.py`. The [quick start](../../../p1-DisorderLatticeProperties/SIMscripts/DAMAGE.md)
has execution commands. No ML loader/head, disordered/bulk processing or job is
included. Only the two authorised periodic Ti B21 samples have been inspected.

Original mesh identities/connectivity/coordinates matched; input SHA-256 values
were unchanged by read-only extraction. Earlier live-ODB verification matched
**296,079 STATUS values**, all-frame parent failure, and **140,990 section SDEG
checks**: [UT verification](UT-periodic-damage.verification.json),
[FT verification](FT-periodic-damage.verification.json). This report rebuild reads
saved NPZs only. [Summary/provenance](results-summary.json), [raw examples](examples.json),
[UT frame table](UT-frame-summary.csv), [FT frame table](FT-frame-summary.csv),
[ODB inventory](odb-inventory.json), and UT/FT-periodic-damage.npz plus adjacent
JSON preserve the underlying evidence. Generated research products remain ignored.

Earlier text-only topology evidence remains in the UT/FT-periodic-strut-map.csv,
topology-summary.json, topology-examples.json and topology-examples.png files;
its odb_opened=false describes that earlier stage. Initial canonical order is
specific to this audit, not existing ML node order. No source artifact changed.
"""
    (output/'README.md').write_text(report)
    (output/'examples.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'results-summary.json').write_text(json.dumps({task:case[2] for task,case in cases.items()},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ut',required=True)
    parser.add_argument('--ft',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    cases={task:load(path,task) for task,path in (('UT',args.ut),('FT',args.ft))}
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    records=[r for a,m,s in cases.values() for r in select_examples(a)]
    maps(cases,output)
    examples_figure(cases,records,output)
    histories(cases,records,output)
    tip_figure(cases['FT'],output)
    write_report(cases,records,output)
    print(json.dumps({task:case[2] for task,case in cases.items()},indent=2))


if __name__=='__main__':
    main()

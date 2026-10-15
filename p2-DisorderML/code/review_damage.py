"""Readable review of two explicit strut-damage NPZs; no ODB/model access."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from resources.damage import require, validate_damage_arrays


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
            status=float(a["status"][frame,index]), damage=float(a["element_damage"][frame,index]) if np.isfinite(a["element_damage"][frame,index]) else None,
            sdeg=[dict(section=int(a["point_section"][p]), integration=int(a["point_integration"][p]),
                       description=str(a["point_section_description"][p]), value=float(a["sdeg"][frame,p])) for p in points]))
    return dict(task=str(a["task"]), kind=kind, strut_id=int(parent), frame=int(frame),
                time=float(a["frame_time"][frame]), loading=float(a["global_loading"][frame]),
                endpoint_labels=a["endpoint_node_label"][parent].tolist(),
                endpoint_coordinates=a["endpoint_xy0"][parent].tolist(), elements=elements,
                damage=float(a["strut_damage"][frame,parent]) if np.isfinite(a["strut_damage"][frame,parent]) else None,
                failed_elements=int(a["failed_element_count"][frame,parent]))


def select_examples(a):
    damage, failed = a["strut_damage"], a["strut_failed"]
    tip = np.flatnonzero((a["canonical_endpoint"] == [381,422]).all(axis=1))
    intact = int(tip[0]) if len(tip) else 0
    records = [parent_record(a, intact, 0, "intact")]
    partial = np.argwhere(a["strut_valid"] & ~failed & (damage > 0) & (damage < 1))
    if len(partial):
        frame,parent = min(partial, key=lambda item:abs(damage[tuple(item)]-.5))
        records.append(parent_record(a,int(parent),int(frame),"partial"))
    elif str(a["partial_rule"]) == "failure-only":
        point_parent=a["parent_strut_index"][a["point_element_index"]]
        active=~failed[:,point_parent] & a["strut_failure_known"][:,point_parent]
        candidates=np.argwhere(active & (a["sdeg"]>0) & (a["sdeg"]<1))
        if len(candidates):
            frame,point=candidates[0]
            records.append(parent_record(a,int(point_parent[point]),int(frame),"partial"))
    one = np.argwhere(failed & (a["failed_element_count"] == 1) & (a["subdivision_count"][None,:] > 1))
    if len(one):
        frame,parent = one[0]
        records.append(parent_record(a,int(parent),int(frame),"one_failed_element"))
    return records


def observed_partial(a):
    """Any reported partial SDEG in an all-active parent; no scalar reduction."""
    result=np.zeros(a['strut_failed'].shape,bool)
    point_parent=a['parent_strut_index'][a['point_element_index']]
    for parent in range(result.shape[1]):
        raw=a['sdeg'][:,point_parent==parent]
        complete=a['element_sdeg_complete'][:,a['parent_strut_index']==parent].all(axis=1)
        result[:,parent]=((raw>0) & (raw<1)).any(axis=1) & complete & a['strut_failure_known'][:,parent] & ~a['strut_failed'][:,parent]
    return result


def segments(a, elements, frame=None):
    lookup = {int(n):i for i,n in enumerate(a["node_label"])}
    indices = np.array([[lookup[int(n)] for n in conn] for conn in a["element_connectivity"][elements]])
    xy = a["node_xy0"][indices]
    return xy if frame is None else xy + a["node_displacement"][frame,indices]


def draw_parent(ax, a, parent, frame, annotate=True):
    members = np.flatnonzero(a["parent_strut_index"] == parent)
    xy = segments(a, members, frame)
    collection = LineCollection(xy, cmap="viridis", norm=Normalize(0,1), linewidths=4)
    collection.set_array(np.ma.masked_invalid(a["element_damage"][frame,members]))
    ax.add_collection(LineCollection(xy,colors='0.65',linewidths=4))
    ax.add_collection(collection)
    broken=a['element_failed'][frame,members]
    if broken.any():
        ax.add_collection(LineCollection(xy[broken],colors='red',linewidths=4.5))
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
    for row,(task,(a,m,_)) in enumerate(cases.items()):
        first=np.flatnonzero(a["strut_failed"].any(axis=1))
        selected=[0,int(first[0]) if len(first) else len(a["frame_time"])//2,len(a["frame_time"])-1]
        for ax,frame in zip(axes[row],selected):
            ax.add_collection(LineCollection(a['endpoint_xy0'],colors='0.8',linewidths=.7))
            collection=LineCollection(a["endpoint_xy0"],cmap="viridis",norm=Normalize(0,1),linewidths=1)
            collection.set_array(np.ma.masked_invalid(a["strut_damage"][frame]))
            ax.add_collection(collection)
            broken=a["strut_failed"][frame]
            if broken.any():
                ax.add_collection(LineCollection(a["endpoint_xy0"][broken],colors="red",linewidths=1.8))
            ax.autoscale_view()
            ax.set_aspect("equal")
            value=a["global_loading"][frame]
            loading=f'nominal strain {value:.4f}' if task=="UT" else f'load-point U2 {value:.3f} mm'
            ax.set(title=f'{task}: frame {frame}, t={a["frame_time"][frame]:.5f} s\n{loading}; failed {broken.sum()}',xlabel="initial x [mm]",ylabel="initial y [mm]")
    fig.colorbar(collection,ax=axes.ravel().tolist(),label="Strut D (red overlay = STATUS-confirmed failed parent)",shrink=.75)
    mode=str(next(iter(cases.values()))[0]['partial_rule'])
    fig.suptitle(f"Real periodic ODB damage: rule {mode}; grey = partial/unavailable numeric D\nUT main body only; initial coordinates",fontsize=13)
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
            values=[e["damage"] for e in r["elements"]]
            arithmetic=("max("+", ".join(f"{v:.3g}" for v in values)+f') = {r["damage"]:.3g}') if all(v is not None for v in values) else ('1: verified failed member' if r['damage']==1 else 'partial aggregation pending')
            endpoints=" -> ".join(f'n{n} {tuple(xy)}' for n,xy in zip(r["endpoint_labels"],r["endpoint_coordinates"]))
            ax.set_title(f'{task} {kind.replace("_"," ")}: strut {r["strut_id"]}, frame {r["frame"]}\nD: {arithmetic}\nInitial endpoints {endpoints}',fontsize=9)
    fig.suptitle("Real element values: raw bottom/top SDEG; S=whole-element STATUS (0 = failure)\nDeformed geometry at true scale; red = failed member; grey = unassigned partial D",fontsize=13)
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
                if str(a['partial_rule'])=='max':
                    axes[row,0].plot(x,a["strut_damage"][:,parent],"o-",markersize=3,label=f'{r["kind"]}: strut {parent}')
                else:
                    axes[row,0].step(x,a['strut_failed'][:,parent].astype(int),where='post',label=f'parent {parent}: failure')
                    if r['kind']=='partial':
                        members=np.flatnonzero(a['parent_strut_index']==parent)
                        element_index=next(index for index in members if (a['sdeg'][r['frame'],a['point_element_index']==index]>0).any())
                        for point in np.flatnonzero(a['point_element_index']==element_index):
                            axes[row,0].plot(x,a['sdeg'][:,point],'o--',markersize=2.5,label=f'e{a["element_label"][element_index]} section{a["point_section"][point]} SDEG')
        axes[row,0].set(xlabel=label,ylabel="Assigned D / raw section SDEG",ylim=(-.03,1.06),title=f"{task}: selected parent histories (21 recorded frames)")
        axes[row,0].legend(fontsize=8)
        axes[row,1].step(x,a["strut_failed"].sum(axis=1),where="post",label="failed parents")
        partial=observed_partial(a).sum(axis=1)
        axes[row,1].plot(x,partial,"o-",markersize=3,label="partially degraded, active parents")
        axes[row,1].set(xlabel=label,ylabel="Number of struts",title=f'{task}: failure and partial degradation counted separately')
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
        value=a['strut_damage'][frame,parent]
        shown=f'{value:.4g}' if np.isfinite(value) else 'unassigned partial'
        ax.set_title(f'FT tip strut {parent}: frame {frame}\nU2={a["global_loading"][frame]:.3f} mm; D={shown}',fontsize=10)
    axes[3].plot(a["global_loading"],a["strut_damage"][:,parent],"o-",label="assigned parent D")
    for index in np.flatnonzero(a["parent_strut_index"]==parent):
        for point in np.flatnonzero(a['point_element_index']==index):
            axes[3].plot(a['global_loading'],a['sdeg'][:,point],'.--',markersize=2,linewidth=.8,label=f'e{a["element_label"][index]} s{a["point_section"][point]}')
    axes[3].set(xlabel="FT load-point U2 [mm]",ylabel="Damage",ylim=(-.02,1.05),title="Parent and constituent histories")
    axes[3].legend(fontsize=7)
    fig.suptitle("Crack-adjacent parent: initial endpoints (120,90) and (120,100); deformed geometry at true scale",fontsize=12)
    fig.savefig(output/"damage-tip-history.png",dpi=160)
    plt.close(fig)


def write_report(cases,records,output):
    rows=[]
    for task,(a,m,summary) in cases.items():
        partial_mask=observed_partial(a)
        counts=dict(zip(*np.unique(a["subdivision_count"],return_counts=True)))
        first=np.flatnonzero(a["strut_failed"].any(axis=1))
        summary.update(first_failure_frame=int(first[0]) if len(first) else -1,
                       first_failure_time=float(a["frame_time"][first[0]]) if len(first) else None,
                       first_failure_loading=float(a["global_loading"][first[0]]) if len(first) else None,
                       final_partial_struts=int(partial_mask[-1].sum()), partial_rule=str(a['partial_rule']),
                       missing_status=int((~a["status_valid"]).sum()),missing_sdeg=int((~a["sdeg_valid"]).sum()),
                       subdivision_histogram={str(k):int(v) for k,v in counts.items()},
                       source_sha256=m["source_sha256"],source_unchanged_sha256=m["source_unchanged_sha256"],
                       status_reactivations=m["status_reactivations"])
        rows.append(f'| {task} | {summary["struts"]:,} | {summary["elements"]:,} | {summary["points"]:,} | {summary["final_failed_struts"]} | {summary["final_partial_struts"]} | {summary["first_failure_frame"]} |')
        with (output/f'{task}-frame-summary.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['frame','time_s','global_loading','reaction_force_N','failed_struts','partial_struts','valid_struts'])
            for j,time in enumerate(a["frame_time"]):
                partial=partial_mask[j]
                writer.writerow([j,time,a["global_loading"][j],a["reaction_force"][j],a["strut_failed"][j].sum(),partial.sum(),a["strut_valid"][j].sum()])
    modes={str(v[0]['partial_rule']) for v in cases.values()}
    report='''# Periodic element-to-strut damage results

Exported partial-rule mode: **'''+', '.join(sorted(modes))+'''**. In failure-only
mode, partial numeric element/strut D stays unassigned (NaN) pending researcher
approval. Raw partial SDEG observations and verified failure remain fully available.
The maximum rule is implemented and synthetically tested, not silently adopted.

Real Abaqus 2024 Python reads of **only the two authorised periodic samples**.
Both have 21 actual recorded frames, including frame 0. UT uses only BodySet;
grips are excluded. All source SHA-256 fingerprints are unchanged before/after
read-only access. Each original INP and ODB agrees in all mesh node/element IDs,
initial coordinates and connectivity; source identity beyond this agreement is
not independently established by an archived run manifest.

| Task | Struts | Elements | Reported SDEG points/frame | Final failed parents | Final active partial parents | First failure frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+"\n".join(rows)+'''

## Exact algorithm

1. Parse the original inline B21 INP mesh and named sets with instance-scoped
   node/element labels. Recognise only the checked periodic FCC profile;
   disordered original-joint identity requires separate validated metadata.
2. Identify original geometric joints, then traverse actual beam connectivity
   from joint to joint. Require unique element coverage, degree-two interior
   nodes, straight monotonic chains and unique unordered endpoint pairs. Measure
   subdivisions; never use consecutive labels or a fixed element count.
3. Match the INP against the ODB mesh and selection. Open with readOnly=True.
   Preserve exact step/frame/increment/time and raw SDEG, DUCTCRT and STATUS.
   Requested DMICRT appears as DUCTCRT; it is initiation, not degradation.
4. These ODBs store SDEG at integration point 1, section points **1 and 5**
   (bottom/top). The available maximum partial rule uses the explicitly selected maximum of
   those **reported** values. This is a severity summary, not cross-section
   average degradation or proof that unreported interior points were observed.
5. Whole-element STATUS=0 establishes a failed element and sets its derived
   D to 1. STATUS=1 is active. SDEG reaches about 0.99999 in these samples;
   exact SDEG==1 would miss failures. Point damage alone never establishes whole
   element failure. Raw values remain unchanged.
6. Set parent D=1 if any member is verified failed; with the maximum rule use maximum member
   severity when complete. In failure-only mode, partial D remains unassigned.
   Missing STATUS/section observations remain unknown,
   and incomplete parents are invalid unless failure is already established.
   This is not a sum or an average; subdivision count does not multiply D.
7. Store observed failure, validity/coverage and first-failure brackets separately.
   No failure carry-forward was enabled for these tests. All observed statuses
   are non-reactivating; deleted elements remain represented by STATUS=0.
8. Write one versioned, non-pickled NPZ per task. Reopen with allow_pickle=False
   and independently recompute the derived arrays from saved raw observations.

## Loading and time

All stored SDEG values in these two ODBs are 0 or 0.9999899864196777. Active
partially damaged elements can have one degraded surface and one undamaged
surface. No intermediate softening ramp is resolved at the recorded frames;
the examples retain those actual values, without inventing smooth damage curves.

The tracked [full algorithm and contract](../../../p1-DisorderLatticeProperties/SIMscripts/DAMAGE.md)
contains execution commands and scope/validation rules. The STATUS/SDEG meanings
are supported by the accessible
[Abaqus documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-damageevolductile.htm);
the actual ODB execution here used Abaqus 2024.

UT plots use actual field-frame mean SET-TOP U2 divided by initial body height
H=190 mm, matching the legacy nominal loading convention. Additional body_strain
stores (mean SET-TOPBODY U2 - mean SET-BOTTOMBODY U2)/H. FT plots use actual LOAD
reference-point U2 in mm. FT U2 is not labelled material strain. Reaction force
is summed RF2 at the corresponding loading set. Original U2/RF2 history times and
values are retained separately; no history interpolation, ordinal joins,
zero-fill or invented uniform frame grid is used.

## Examples and figures

Each example in examples.json lists original endpoints/coordinates, exact raw
section SDEG and STATUS per element, reduction arithmetic and resulting parent D.
Partial observations, intact and one-failed-element states are selected from actual records;
missing states would be reported, never fabricated.

![Parent damage snapshots](damage-overview.png)

![Selected raw element examples](damage-examples.png)

![Damage versus explicitly defined loading](damage-histories.png)

![Crack-adjacent parent over several frames](damage-tip-history.png)

## NPZ contract and limits

schema_version=strut_damage_v1. Native task struts are the entity axis; canonical
endpoint indices refer to this audit's periodic (y,x)-sorted 800-node grid, not
existing saved ML array order. canonical_node_present records initial FT absence;
native present struts have strut_present=True. Missing initial FT struts are not
failed struts. No node damage labels, damage head, changing inference edges or
training integration were added.

The NPZ stores instance-scoped element/endpoint labels, original connectivity,
parent membership/order, initial coordinates, reported section/IP identities,
raw SDEG/DUCTCRT/STATUS and validity, field-frame nodal displacements, actual
loading/time, raw history records, derived element/strut values, coverage, failure
and onset brackets, explicit reduction rules, and JSON provenance/code hashes.
First failure is observed on the field-output grid; its exact onset between
frames is unresolved. The successful tests apply to these two periodic Ti B21
files, not all archived/disordered cases. No source artifact was modified.

Raw-data coverage and numeric-label coverage are separate. In failure-only mode,
all expected raw observations may be present while a partial numeric label is
unassigned by policy. strut_observation_coverage records raw availability;
strut_coverage records assigned member D coverage.

The independent live-ODB verification files UT-periodic-damage.verification.json
and FT-periodic-damage.verification.json record exact matches for all 296,079
STATUS values/all-frame parent failures and 140,990 section SDEG spot checks.

## Supporting evidence

The retained initial INP-only topology stage is separate from the current ODB
results: [UT parent map](UT-periodic-strut-map.csv),
[FT parent map](FT-periodic-strut-map.csv), [initial summary](topology-summary.json),
[initial examples](topology-examples.json) and [labelled topology](topology-examples.png).
The initial summary's odb_opened=false describes that earlier text-only pass;
current ODB evidence is in the periodic damage NPZ/JSONs and results-summary.json.
[ODB inventory](odb-inventory.json) records actual field/history names and probes.

Current products: UT-periodic-damage.npz/.json, FT-periodic-damage.npz/.json,
UT-frame-summary.csv, FT-frame-summary.csv, examples.json, results-summary.json
and the four figures above. Producer/review code is tracked outside samples;
generated research artifacts remain ignored under the existing repository policy.

Researcher selection context (not processed here): D denotes highest UT ductility,
K initial FT fracture toughness, and B a same-disorder UT/FT pair maximising the
UT-FT property Pareto front. Keep these meanings distinct from damage D.
'''
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

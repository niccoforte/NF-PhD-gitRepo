"""Readable review of two explicit strut-damage NPZs; no ODB/model access."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from resources.damage import require, validate_damage_arrays


FAILED_COLOR = "#d62728"
PENDING_COLOR = "0.6"


def damage_scale(mode):
    """The numeric legend includes the same red used for D=1 on the mesh."""
    if mode == "failure-only":
        return ListedColormap(["#440154", FAILED_COLOR]), BoundaryNorm([-.5, .5, 1.5], 2)
    return LinearSegmentedColormap.from_list(
        "damage", ["#440154", "#31688e", "#35b779", "#fde725", FAILED_COLOR]), Normalize(0, 1)


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
    partial = np.argwhere(a["strut_valid"] & ~failed & (damage > 0))
    if len(partial):
        frame,parent = min(partial, key=lambda item:abs(damage[tuple(item)]-.5))
        records.append(parent_record(a,int(parent),int(frame),"partial"))
    elif str(a["partial_rule"]) == "failure-only":
        point_parent=a["parent_strut_index"][a["point_element_index"]]
        active=~failed[:,point_parent] & a["strut_failure_known"][:,point_parent]
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
    cmap, norm = damage_scale(str(a["partial_rule"]))
    collection = LineCollection(xy, cmap=cmap, norm=norm, linewidths=4)
    collection.set_array(np.ma.masked_invalid(a["element_damage"][frame,members]))
    unassigned = ~np.isfinite(a["element_damage"][frame,members])
    pending = unassigned & a["element_sdeg_complete"][frame,members] & (a["status"][frame,members] == 1)
    ax.add_collection(LineCollection(xy[pending],colors=PENDING_COLOR,linewidths=4))
    ax.add_collection(LineCollection(xy[unassigned & ~pending],colors='black',linestyles='dashed',linewidths=4))
    ax.add_collection(collection)
    broken=a['element_failed'][frame,members]
    if broken.any():
        if str(a["partial_rule"]) == "max":
            ax.add_collection(LineCollection(xy[broken],colors='black',linewidths=7))
        ax.add_collection(LineCollection(xy[broken],colors=FAILED_COLOR,linewidths=4.5))
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
    modes = {str(case[0]["partial_rule"]) for case in cases.values()}
    require(len(modes) == 1, "Use matching partial rules for a shared map legend")
    mode = next(iter(modes))
    cmap, norm = damage_scale(mode)
    has_pending = has_missing = False
    for row,(task,(a,m,_)) in enumerate(cases.items()):
        partial = observed_partial(a)
        first=np.flatnonzero(a["strut_failed"].any(axis=1))
        selected=[0,int(first[0]) if len(first) else len(a["frame_time"])//2,len(a["frame_time"])-1]
        for ax,frame in zip(axes[row],selected):
            unassigned = ~a["strut_valid"][frame]
            pending = unassigned & partial[frame]
            missing = unassigned & ~pending
            has_pending |= bool(pending.any())
            has_missing |= bool(missing.any())
            ax.add_collection(LineCollection(a["endpoint_xy0"][pending],colors=PENDING_COLOR,linewidths=1))
            ax.add_collection(LineCollection(a["endpoint_xy0"][missing],colors="black",linestyles="dashed",linewidths=1))
            collection=LineCollection(a["endpoint_xy0"],cmap=cmap,norm=norm,linewidths=1)
            collection.set_array(np.ma.masked_invalid(a["strut_damage"][frame]))
            ax.add_collection(collection)
            broken=a["strut_failed"][frame]
            if broken.any():
                if mode == "max":
                    ax.add_collection(LineCollection(a["endpoint_xy0"][broken],colors="black",linewidths=3.3))
                ax.add_collection(LineCollection(a["endpoint_xy0"][broken],colors=FAILED_COLOR,linewidths=1.8))
            ax.autoscale_view()
            ax.set_aspect("equal")
            value=a["global_loading"][frame]
            loading=f'nominal strain {value:.4f}' if task=="UT" else f'load-point U2 {value:.3f} mm'
            ax.set(title=f'{task}: frame {frame}, t={a["frame_time"][frame]:.5f} s\n{loading}; failed {broken.sum()}',xlabel="initial x [mm]",ylabel="initial y [mm]")
    bar_label = "Assigned strut D; red D=1 = verified failure" if mode == "failure-only" else "Strut D: maximum reported severity (red at 1)"
    bar = fig.colorbar(collection,ax=axes.ravel().tolist(),label=bar_label,shrink=.75)
    if mode == "failure-only":
        bar.set_ticks([0, 1], labels=["0: no reported damage", "1: verified failed"])
    legend = []
    if mode == "max":
        legend.append(Line2D([], [], color="black", linewidth=4,
            label="Black outline: STATUS-confirmed failure (severity alone does not establish failure)"))
    if has_pending:
        legend.append(Line2D([], [], color=PENDING_COLOR, linewidth=3,
            label="Active with surface damage; strut D awaiting reduction rule"))
    if has_missing:
        legend.append(Line2D([], [], color="black", linestyle="--", linewidth=2,
            label="Missing observations; strut D unknown"))
    if legend:
        fig.legend(handles=legend,loc="outside lower center",fontsize=10)
    state_note = "grey is active surface damage awaiting scalar reduction" if mode == "failure-only" else "colour is severity; black outline confirms failure"
    fig.suptitle(f"Periodic UT/FT: initial geometry, recorded damage states ({mode})\nUT main body only; {state_note}",fontsize=13)
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
    state_note = "red = failed member; grey = active surface damage awaiting scalar rule" if str(next(iter(cases.values()))[0]['partial_rule']) == 'failure-only' else "colour = severity; black outline = verified failed member"
    fig.suptitle("Real element values: raw bottom/top SDEG; S=whole-element STATUS (0 = failure)\nTrue-scale deformation; " + state_note,fontsize=13)
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
    policy = ("**Current choice: failure-only.** Grey struts are active and have observed surface "
              "damage, but their single numeric D awaits a reduction rule. Their data is present. "
              "The implemented maximum rule has not been adopted.") if modes == {'failure-only'} else (
              "**Saved choice: " + ', '.join(sorted(modes)) + ".** Maximum mode summarises the "
              "largest reported surface/element degradation; whole-element failure remains a separate flag.")
    ut, ft = cases['UT'][2], cases['FT'][2]
    appearance = ("| Purple, D=0 | No reported surface damage; all member elements are active. |\n"
        "| Grey, no scalar D | Surface damage is observed, but every member is active; scalar reduction awaits a choice. |\n"
        "| Red, D=1 | A member has STATUS=0; the parent is classified failed. |") if modes == {'failure-only'} else (
        "| Continuous colour, D=0..1 | Maximum reported local severity, not average strut degradation. |\n"
        "| Black outline | A member has STATUS=0; the parent is classified failed. Colour alone does not establish failure. |")
    example = next((r for r in records if r['task'] == 'UT' and r['kind'] == 'partial'), None)
    if example:
        damaged = next(e for e in example['elements'] if any(p['value'] > 0 for p in e['sdeg']))
        raw = ', '.join(f"section {p['section']}: {p['value']:.10g}" for p in damaged['sdeg'])
        severity = max(p['value'] for e in example['elements'] for p in e['sdeg'])
        assigned = ('**unassigned (NaN)** pending the scalar reduction choice' if example['damage'] is None
                    else f"**{example['damage']:.10g}** under the saved maximum rule")
        example_text = (f"A real UT example is **strut {example['strut_id']}, frame {example['frame']}**. "
            f"Element **{damaged['label']}** reports {raw} and STATUS **{damaged['status']:g} (active)**.\n\n"
            f"Its failure state is known: **not failed**. Its current scalar D is {assigned}. "
            f"The maximum of all reported surface/element values is **{severity:.10g}**; "
            "using that rule preserves **failed=False**. It measures worst reported local damage, "
            "not average strut stiffness loss. Exact constituent values appear below and in examples.json.")
    else:
        example_text = 'No active surface-damage example was observed for UT at the recorded frames.'
    raw_values = np.unique(np.concatenate([case[0]['sdeg'][case[0]['sdeg_valid']] for case in cases.values()]))
    raw_text = 'Stored SDEG values: ' + ', '.join(f'{v:.16g}' for v in raw_values) + '.' if len(raw_values) <= 8 else 'Raw section histories are retained in the NPZs.'
    report='''# Periodic element-to-strut damage results

**The exporter works on the two authorised periodic ODBs.** At the final recorded
frame, UT has **'''+str(ut['final_failed_struts'])+''' verified failed struts** and FT **'''+str(ft['final_failed_struts'])+'''**.
Active struts with observed surface damage: UT **'''+str(ut['final_partial_struts'])+'''**, FT **'''+str(ft['final_partial_struts'])+'''**.

'''+policy+'''

![Full-body damage states with red D=1 on the legend](damage-overview.png)

| Appearance | Meaning under the saved rule |
| --- | --- |
'''+appearance+'''
| Black dashed, if present | Observations missing; scalar D unknown. |

The numeric scale includes **red at D=1**. Grey, when present, has a separate
state key because it is not a numeric damage value. In maximum mode, colour is
severity and a **black outline** independently identifies STATUS-confirmed failure.

## What the real data shows

| Task | Struts | Elements | Reported SDEG points/frame | Final failed parents | Final active partial parents | First failure frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+"\n".join(rows)+'''

Recorded frames including frame 0: UT **'''+str(ut['frames'])+'''**, FT **'''+str(ft['frames'])+'''**.
UT uses BodySet only; grips are excluded. Subdivision counts are measured,
not assumed: UT '''+', '.join(ut['subdivision_histogram'])+'''; FT '''+', '.join(ft['subdivision_histogram'])+'''.
Missing expected STATUS/SDEG records: UT '''+str(ut['missing_status'])+'/'+str(ut['missing_sdeg'])+''';
FT '''+str(ft['missing_status'])+'/'+str(ft['missing_sdeg'])+'''. STATUS reactivations:
UT '''+str(ut['status_reactivations'])+''', FT '''+str(ft['status_reactivations'])+'''. Failure is resolved only on the recorded frame grid.

### Why observed surface damage is not whole-element failure

'''+example_text+'''

'''+raw_text+''' **STATUS=0 identifies whole-element failure.**
Never use exact SDEG=1 as a failure test; the original verification checks failure
against STATUS independently. In these samples, failed elements store SDEG below 1.

![Real intact, active damaged, and one-failed-element examples](damage-examples.png)

If a task has no active surface-damage example at a recorded frame, the figure
states that absence explicitly. [Raw examples](examples.json) retain
original endpoints, element labels/connectivity, exact section values and STATUS.

## Evolution under loading

![Recorded damage/failure versus loading](damage-histories.png)

UT horizontal axis: actual field-frame mean SET-TOP U2 / initial body height
**190 mm** (nominal strain). The NPZ also stores relative body strain from the
top/bottom body interfaces. FT horizontal axis: actual LOAD reference-point
**U2 in mm**, not material strain. Markers are recorded observations; the
intermediate softening path is unresolved.

![FT crack-adjacent parent and constituent histories](damage-tip-history.png)

## Use the implementation

Reusable code is tracked outside samples: p1 `A2_DamageOUTpostProcess.py` reads
one explicit ODB/INP pair, `resources/damage.py` maps/reduces/validates it, and p2
`code/review_damage.py` builds this report from saved NPZs. The
[quick start and optional technical reference](../../../p1-DisorderLatticeProperties/SIMscripts/DAMAGE.md)
contains runnable commands. There is **no DATA/MODEL loader or damage head yet**;
model integration, disordered processing and bulk extraction remain out of scope.

## Validation and provenance

The two original INP/ODB meshes agree in node/element identities, coordinates and
connectivity. Source SHA-256 fingerprints are unchanged by the Abaqus 2024
read-only exports. Independent live-ODB verification matched **296,079 STATUS
values**, all-frame parent failure flags, and **140,990 section-SDEG checks**;
see [UT verification](UT-periodic-damage.verification.json) and
[FT verification](FT-periodic-damage.verification.json).

Supporting products: [summary and fingerprints](results-summary.json),
[UT frame table](UT-frame-summary.csv), [FT frame table](FT-frame-summary.csv),
[ODB field/history inventory](odb-inventory.json), UT/FT-periodic-damage.npz and
adjacent provenance JSON. NPZs retain raw section/IP SDEG, DUCTCRT initiation,
whole-element STATUS, actual loading/time and histories, source/code hashes,
connectivity, failure/observation/label masks and onset brackets. Missing output
is never zero-filled. No failure carry-forward was enabled.

`strut_damage_v1` uses native struts; its audit-canonical endpoint order is not
existing ML node order. Initial absent FT struts are not failed struts. These two
periodic Ti B21 files do not validate arbitrary disordered/archived cases. There
is no independent archived run manifest proving source identity beyond the
mesh agreement. Inputs and existing displacement/training artifacts are unchanged.

Earlier INP-only evidence remains separate: [UT map](UT-periodic-strut-map.csv),
[FT map](FT-periodic-strut-map.csv), [topology summary](topology-summary.json),
[topology examples](topology-examples.json) and [topology figure](topology-examples.png).
The earlier summary's `odb_opened=false` describes that earlier text-only stage.
Generated research products stay ignored; their executable producers are tracked.
D (highest UT ductility), K (initial FT toughness) and B (same-disorder UT/FT
property-Pareto pair) identify other researcher-selected specimens, not processed here.
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

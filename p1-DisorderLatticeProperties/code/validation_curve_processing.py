"""Read-only curve-processing qualification pilot; never rewrites MLdata.

Population transfer CSVs omit time and can contain artificial tails. Their
sample-domain filter sensitivities are not source-qualified modal cutoffs.
Stable regression is a numerical screen, not proof of an elastic body modulus.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from resources.calculations import smooth


TOPOLOGIES = {"tri": 30, "kagome": 20, "hex": 20, "FCC": 20}
TOPOLOGY_COLORS = {"tri": "#ff0000", "kagome": "#000000", "hex": "#008000", "FCC": "#0000ff"}
IDS = (1, 50, 500, 1000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sqrt(np.mean((y-slope*x-intercept)**2))/np.ptp(y))
    return float(slope), float(intercept), residual


def initial_fit(x, y, max_strain=.005, residual_limit=.02, slope_limit=.02):
    """Find earliest three consecutive stable initial windows, with an intercept.

    Start indices 0/1/2 test startup sensitivity. Every chosen end must have
    span, positive slope and a low span-normalised residual, and every tested
    slope must lie within the declared stability limit. No averaging can rescue
    inconsistent windows. The strain cap is an explicit pilot assumption.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    candidates = []
    if len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "insufficient_finite_samples", "candidates": []}
    if np.any(np.diff(x) <= 0):
        return {"status": "nonmonotone_initial_strain", "candidates": []}
    cap = np.flatnonzero(x-x[0] <= max_strain)
    end_max = min(int(cap[-1]), 19) if len(cap) else 0
    for end in range(7, end_max+1):
        slopes, residuals = [], []
        for start in range(3):
            xx, yy = x[start:end+1], y[start:end+1]
            if len(xx) < 6 or np.ptp(yy) <= 0 or np.ptp(xx) <= 0:
                break
            slope, intercept, residual = fit_line(xx, yy)
            slopes.append(slope)
            residuals.append(residual)
        if len(slopes) != 3:
            continue
        median = float(np.median(slopes))
        spread = float(max(abs(np.asarray(slopes)/median-1))) if median > 0 else None
        candidates.append({"end_index": end, "end_strain": float(x[end]),
                           "slope": median, "startup_slopes": slopes,
                           "max_residual_over_span": max(residuals),
                           "startup_relative_slope_spread": spread,
                           "passes": bool(median > 0 and max(residuals) <= residual_limit and spread <= slope_limit)})
        if len(candidates) < 3:
            continue
        group = candidates[-3:]
        slopes = np.array([row["slope"] for row in group])
        stability = float(max(abs(slopes/np.median(slopes)-1)))
        if all(row["passes"] for row in group) and stability <= slope_limit:
            # Choose the central qualified interval, not an average of windows.
            chosen = dict(group[1])
            return {"status": "numerically_stable_candidate", "selected": chosen,
                    "neighbour_window_relative_slope_spread": stability,
                    "material_elasticity_verified": False, "body_gauge_verified": False,
                    "candidates": candidates}
    return {"status": "unresolved_initial_window", "selected": None, "candidates": candidates}


def paired_filter(x, y, corner):
    """Second-order Butterworth, forward/backward on observed sample positions.

    corner is a fraction of sample Nyquist, not a frequency inferred from strain.
    Missing values/gaps must be split before calling. No STATUS interpolation.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) <= 12 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("A finite paired observed prefix of at least 13 samples is required")
    if not 0 < corner < 1:
        raise ValueError("Corner must lie inside sample Nyquist")
    sos = butter(2, corner, output="sos")
    return sosfiltfilt(sos, x), sosfiltfilt(sos, y)


def periodic_causal_filter(time, x, y, natural_frequency_Hz):
    """Hsieh cutoff rule with our specimen's periodic frequency, causal/DC SOS.

    Order two and DC initialisation are explicit Abaqus-default reconstruction
    choices, not filter settings disclosed by Hsieh. No missing data, event or
    loading-axis resampling is created. Above-Nyquist cutoffs fail explicitly.
    """
    from validation_calculations import filter_force
    if not np.isfinite(natural_frequency_Hz) or natural_frequency_Hz <= 0:
        raise ValueError("A positive source-qualified periodic frequency is required")
    xx, info = filter_force(time, x, 2*natural_frequency_Hz, order=2, zero_phase=False)
    yy, _ = filter_force(time, y, 2*natural_frequency_Hz, order=2, zero_phase=False)
    info.update(natural_frequency_Hz=float(natural_frequency_Hz), startup="DC",
                policy="same source-qualified periodic cutoff for matching ordered/full/mechanism specimens",
                histories="native timestamps and observations retained; no missing tail filled",
                order_basis="Abaqus default reconstruction; Hsieh does not report order")
    return (xx, yy), info


def periodic_cutoff_registry(root, accept_physical_modes=False):
    """Bind eight family cutoffs after numerical screen and mode-shape review."""
    modal = Path(root)/"audit/modal/production-periodic"
    records = []
    for topology, nx in TOPOLOGIES.items():
        for task in ("UT", "FT"):
            folder = modal/("pm_{}_{}".format(topology.lower(), task.lower()))
            path = folder/"modes.json"
            if not path.exists():
                records.append(dict(topology=topology, task=task, status="modal_output_pending"))
                continue
            data = json.loads(path.read_text())
            selected = data.get("candidate_mode")
            if not selected or not (folder/"mode_shapes.png").is_file():
                raise ValueError("Physical-mode screen/plot unresolved: " + str(folder))
            if sha(data["source_inp"]) != data["source_inp_sha256"]:
                raise ValueError("Periodic source changed")
            records.append(dict(topology=topology, task=task, nnx=nx, l_mm=10., nominal_relative_density=.2,
                status="accepted_physical_mode" if accept_physical_modes else "requires_mode_shape_review",
                mode=selected["mode"], natural_frequency_Hz=selected["frequency_Hz"],
                cutoff_Hz=2*selected["frequency_Hz"], boundary=data["boundary"],
                modal_record=str(path), modal_record_sha256=sha(path),
                mode_shape_review=("Mode deformation, spatial support and constraints visually inspected" if accept_physical_modes else None),
                source_inp=data["source_inp"], source_inp_sha256=data["source_inp_sha256"],
                application="all IDs of matched periodic/full/mechanism family; no disorder-specific eigenproblems"))
    result = dict(policy="Hsieh real-time Butterworth fc=2fn; periodic-only frequencies", order=2,
                  startup="DC", zero_phase=False, families=records,
                  accepted_periodic_cutoff_as_shared_disorder_approximation=True,
                  scope="10mm20%; tri nx30; kagome/hex/FCC nx20; task and boundary specific",
                  limits=["No inference that disordered frequencies equal periodic frequencies",
                          "Saved history filtering cannot undo aliasing before output decimation",
                          "Cutoffs at/above saved-history Nyquist are explicitly unresolved, never silently capped"])
    out = Path(root)/"audit/curve-processing/periodic-cutoffs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    return result


def extract_periodic_cutoff_examples(root, archive_root, data_root):
    """Abaqus read-only recovery of24 paired histories, no solver/submission.

    Reuses each existing transfer's normalisation after checking it against
    genuine ODB response. Final native samples are recovered rather than padded.
    """
    from odbAccess import openOdb
    from validation_history_audit import field_summary
    archive_root, data_root = Path(archive_root), Path(data_root)
    out = Path(root)/"audit/curve-processing/periodic-cutoff-native"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for topology, nx in TOPOLOGIES.items():
        for task, label in (("UT", "Ductile"), ("FT", "Fracture")):
            for design, group, transfer_group, token in (
                    ("periodic", "per/0.0", "per", "per"),
                    ("full", "disNodes/0.2", "disNodes/0.2", "20disNodes-lhs-all"),
                    ("mechanism", "disNodes/Target-xs/0.2", "disNodes/Target-xs/0.2", "20disNodes-lhs-xs")):
                stem = "{}-{}-{}-{}-1".format(label, topology, nx, token)
                base = archive_root/group/topology
                matches = [job/"zip"/(stem+".inp") for job in base.iterdir()
                           if (job/"zip"/(stem+".inp")).is_file()]
                if len(matches) != 1:
                    raise ValueError("Ambiguous/missing native example: " + stem)
                inp, transfer = matches[0], data_root/transfer_group/topology/"transfer"/("OUT-"+stem+".csv")
                source, before = inp.with_suffix(".odb"), sha(inp.with_suffix(".odb"))
                odb = openOdb(str(source), readOnly=True)
                try:
                    if len(odb.steps) != 1:
                        raise ValueError("Expected one original nonlinear step")
                    step = list(odb.steps.values())[0]
                    if task == "FT":
                        regions = [step.historyRegions["Node ASSEMBLY.1"]]
                    else:
                        nodes = [node for group_nodes in odb.rootAssembly.nodeSets["SET-TOP"].nodes for node in group_nodes]
                        regions = []
                        for node in nodes:
                            matching = [r for name,r in step.historyRegions.items()
                                        if name.startswith("Node PART-1-1.") and name.endswith("."+str(node.label))
                                        and "U2" in r.historyOutputs and "RF2" in r.historyOutputs]
                            if len(matching) != 1:
                                raise ValueError("Ambiguous exact-source clamp node history")
                            regions.append(matching[0])
                    time = np.asarray([t for t,v in regions[0].historyOutputs["U2"].data])
                    displacement, force = [], []
                    for region in regions:
                        for key, target in (("U2", displacement), ("RF2", force)):
                            pairs = np.asarray(region.historyOutputs[key].data)
                            if not np.array_equal(pairs[:,0], time):
                                raise ValueError("Native force/displacement grids differ")
                            target.append(pairs[:,1])
                    x_native, y_native = np.mean(displacement, axis=0), np.sum(force, axis=0)
                    status_event = None
                    for frame in step.frames:
                        if "STATUS" in frame.fieldOutputs and field_summary(frame.fieldOutputs["STATUS"])["min"] == 0:
                            status_event = float(frame.frameValue)
                            break
                finally:
                    odb.close()
                if before != sha(source):
                    raise ValueError("Read-only extraction changed source")
                old = np.loadtxt(str(transfer), delimiter=",", usecols=(0,1))
                common = min(len(time)-1, len(old))
                scales, errors = [], []
                for raw, previous in ((x_native, old[:,0]), (y_native, old[:,1])):
                    eligible = (np.abs(raw[:common]) > np.max(np.abs(raw[:common]))*1e-7) & (previous[:common] != 0)
                    scale = float(np.median(previous[:common][eligible]/raw[:common][eligible]))
                    error = float(np.max(np.abs(previous[:common][eligible]-raw[:common][eligible]*scale))/np.max(np.abs(previous[:common][eligible])))
                    if error > 2e-5:
                        raise ValueError("Native/retained transfer normalisation differs: " + stem)
                    scales.append(scale)
                    errors.append(error)
                x, y = x_native*scales[0], y_native*scales[1]
                if not all(np.isfinite(v).all() for v in (time,x,y)) or np.any(np.diff(time) <= 0):
                    raise ValueError("Finite strictly increasing native history required")
                destination = out/("{}_{}_{}.npz".format(topology,task,design))
                np.savez_compressed(str(destination), time=time, x=x, y=y)
                row = dict(topology=topology, task=task, design=design, id=0 if design=="periodic" else 1,
                           source_odb=str(source), source_odb_sha256=before, source_inp=str(inp), source_inp_sha256=sha(inp),
                           transfer=str(transfer), transfer_sha256=sha(transfer), native=str(destination), native_sha256=sha(destination),
                           samples=len(time), first_STATUS_deletion_time_s=status_event, time_start_s=float(time[0]),
                           time_last_s=float(time[-1]), normalisation_factors=scales,
                           native_transfer_relative_errors=errors, source_unchanged=True,
                           observed_support_only=True, no_missing_samples_manufactured=True)
                records.append(row)
                (out/"index.json").write_text(json.dumps(records, indent=2)+"\n")
                print(topology,task,design,len(time),flush=True)
    return records


def apply_periodic_examples(root):
    """Apply accepted periodic cutoffs to genuine ID1 histories, preserve sources."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    root = Path(root)
    folder = root/"audit/curve-processing"
    registry_path = folder/"periodic-cutoffs.json"
    registry = json.loads(registry_path.read_text())
    controls = {(r['topology'],r['task']):r for r in registry['families']}
    source_index = folder/"periodic-cutoff-native/index.json"
    records = json.loads(source_index.read_text())
    out = folder/"filtered-curves"
    out.mkdir(parents=True,exist_ok=True)
    results, figures = [], []
    for task, stem in (("UT","uat-periodic-cutoff"),("FT","ft-periodic-cutoff")):
        fig,axes=plt.subplots(4,3,figsize=(12.8,11.5),squeeze=False)
        for row,topology in enumerate(TOPOLOGIES):
            control=controls[(topology,task)]
            if control['status'] != 'accepted_physical_mode':
                raise ValueError('Periodic physical mode has not been reviewed')
            for col,design in enumerate(('periodic','full','mechanism')):
                record=next(r for r in records if r['topology']==topology and r['task']==task and r['design']==design)
                if sha(record['native']) != record['native_sha256']:
                    raise ValueError('Extracted native observations changed')
                native=np.load(record['native'])
                time,x,y=native['time'],native['x'],native['y']
                nyquist=(len(time)-1)/(2*(time[-1]-time[0]))
                event=record['first_STATUS_deletion_time_s']
                endpoint=(max(1,int(np.searchsorted(time,event,side='left'))-1) if task=='FT' and event is not None else
                          first_crossing(y,.25) if task=='UT' else len(y)-1)
                endpoint=len(y)-1 if endpoint is None else endpoint
                methods=[('native',x,y),('previous_twice_moving_average',x,np.asarray(smooth(smooth(y)))),
                         ('previous_0p4_Nyquist_zero_phase',*paired_filter(x,y,.4))]
                try:
                    (xx,yy),information=periodic_causal_filter(time,x,y,control['natural_frequency_Hz'])
                    methods.append(('periodic_causal_2fn',xx,yy))
                    status='applied'
                    reason=None
                except ValueError as error:
                    status='unresolved_at_saved_sampling'
                    information=None
                    reason=str(error)
                axis=axes[row,col]
                scale=1. if task=='UT' else .001
                styles={'native':dict(color='.70',linewidth=.8,zorder=1),
                        'previous_twice_moving_average':dict(color='.18',linestyle='--',linewidth=1.,zorder=3),
                        'previous_0p4_Nyquist_zero_phase':dict(color='#9467bd',linestyle=':',linewidth=1.25,zorder=4),
                        'periodic_causal_2fn':dict(color=TOPOLOGY_COLORS[topology],linewidth=1.35,zorder=5)}
                comparison=[]
                for method,xx,yy in methods:
                    axis.plot(xx,yy*scale,**styles[method])
                    metrics=numeric_metrics(xx,yy,endpoint)
                    comparison.append(dict(method=method,**metrics))
                    results.append(dict(topology=topology,task=task,design=design,id=record['id'],
                        method=method,cutoff_Hz=control['cutoff_Hz'] if method=='periodic_causal_2fn' else None,
                        natural_frequency_Hz=control['natural_frequency_Hz'],saved_nyquist_Hz=float(nyquist),
                        periodic_filter_status=status,native_samples=len(time),fixed_endpoint_index=endpoint,
                        fixed_endpoint_time_s=float(time[endpoint]),**metrics))
                note=(r'$2f_n$ = '+format(control['cutoff_Hz'],'.1f')+' Hz')
                if status!='applied':
                    note+='\nAbove saved Nyquist; unapplied'
                axis.text(.98,.96,note,ha='right',va='top',transform=axis.transAxes,fontsize=8,
                          bbox=dict(facecolor='white',alpha=.82,edgecolor='none',pad=1.5))
                axis.grid(alpha=.18)
                axis.tick_params(labelsize=8)
                axis.set_ylabel((topology.capitalize() if topology!='FCC' else topology)+'\n'+
                                (r'$\Sigma$ (MPa)' if task=='UT' else 'Force (kN)'),fontsize=9)
                if row==0:
                    axis.set_title({'periodic':'Periodic','full':'Full disorder, ID 1','mechanism':'Mechanism-based, ID 1'}[design],
                                   color={'periodic':'.1','full':'#ff7f0e','mechanism':'#1f77b4'}[design],fontsize=11)
                record=dict(record,periodic_control=control,filter_status=status,reason=reason,
                            filter_information=information,saved_nyquist_Hz=float(nyquist),comparisons=comparison)
        legend=[Line2D([],[],color='.7',lw=1,label='Native response'),
                Line2D([],[],color='.2',lw=1.5,label=r'Causal Butterworth, $2f_n$ (where resolved)'),
                Line2D([],[],color='.18',ls='--',label='Previous twice-moving average'),
                Line2D([],[],color='#9467bd',ls=':',label='Previous 0.4-Nyquist filter')]
        fig.legend(handles=legend,loc='upper center',ncol=2,fontsize=9,frameon=False,bbox_to_anchor=(.5,1.))
        fig.supxlabel(r'Macroscopic strain, $\varepsilon_L$' if task=='UT' else 'Load-line displacement (mm)',fontsize=11)
        fig.tight_layout(rect=(0,.025,1,.95),h_pad=1.4,w_pad=1.7)
        for extension in ('png','svg'):
            path=out/(stem+'.'+extension)
            fig.savefig(str(path),dpi=190)
            figures.append(str(path))
        plt.close(fig)
    with (out/'periodic-cutoff-metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report=dict(periodic_registry=str(registry_path),registry_sha256=sha(registry_path),
                native_index=str(source_index),native_index_sha256=sha(source_index),
                figures=figures,measurements=results,
                cutoff_source='Only the eight exact population-periodic specimens; same cutoff used for every matching design',
                event_handling='Native STATUS event remains unfiltered. Fixed endpoint uses last native sample before STATUS for these curve comparisons, not a replacement material-initiation definition.',
                axis_support='Every point retains its actual native timestamp and paired response; no stretching, padding or artificial tail',
                limitation='Six of eight physical cutoffs exceed saved response Nyquist; no physical filter is applied to those families. Previous0.4-Nyquist and moving averages remain sensitivity comparisons, not Hsieh-calibrated results.',
                methods='Physical filter: causal order2 DC-start Butterworth at2fn; previous0.4: order2 forward/backward; previousMA: twice resources.smooth',
                production_data_unchanged=True)
    (out/'periodic-cutoff-comparison.json').write_text(json.dumps(report,indent=2)+"\n")
    return report


def paired_gallery_selection(candidates, seed=20261110):
    """Response-independent stratified sampling; the same pair serves UT/FT."""
    import random
    rng = random.Random(seed)
    selected = []
    for topology in TOPOLOGIES:
        for design, count in (("full", 4), ("mechanism", 3)):
            eligible = sorted(set(int(value) for value in candidates[topology][design]
                                  if 1 <= int(value) <= 1000))
            if len(eligible) < count:
                raise ValueError("Insufficient paired sources for " + topology + "/" + design)
            for identifier in sorted(rng.sample(eligible, count)):
                selected.append(dict(topology=topology, design=design, id=identifier))
    return selected


def prepare_paired_gallery(root, archive_root, data_root, seed=20261110):
    """Freeze seven paired architecture IDs per topology before reading responses."""
    import re
    archive_root, data_root = Path(archive_root), Path(data_root)
    folder = Path(root)/"audit/curve-processing/paired-examples"
    folder.mkdir(parents=True, exist_ok=True)
    existing = json.loads((Path(root)/"audit/curve-processing/periodic-cutoff-native/index.json").read_text())
    periodic = [row for row in existing if row["design"] == "periodic"]
    candidates, sources, census = {}, {}, []
    for topology, nx in TOPOLOGIES.items():
        candidates[topology] = {}
        for design, group, token in (("full", "disNodes/0.2", "20disNodes-lhs-all"),
                                      ("mechanism", "disNodes/Target-xs/0.2", "20disNodes-lhs-xs")):
            base = archive_root/group/topology
            found = {}
            pattern = re.compile(r"^(Ductile|Fracture)-"+re.escape(topology)+"-"+str(nx)+"-"+re.escape(token)+r"-(\d+)\.inp$")
            for job in sorted(base.iterdir()):
                zip_folder = job/"zip"
                if not zip_folder.is_dir():
                    continue
                for inp in zip_folder.glob("*-"+topology+"-"+str(nx)+"-"+token+"-*.inp"):
                    match = pattern.fullmatch(inp.name)
                    if not match or not 1 <= int(match[2]) <= 1000:
                        continue
                    transfer = data_root/group/topology/"transfer"/("OUT-"+inp.stem+".csv")
                    if inp.with_suffix(".odb").is_file() and transfer.is_file():
                        task = "UT" if match[1] == "Ductile" else "FT"
                        found.setdefault(int(match[2]), {}).setdefault(task, []).append((inp, transfer))
            eligible = [identifier for identifier, values in found.items()
                        if set(values) == {"UT", "FT"} and all(len(items) == 1 for items in values.values())]
            candidates[topology][design] = eligible
            for identifier in eligible:
                for task in ("UT", "FT"):
                    inp, transfer = found[identifier][task][0]
                    sources[topology, design, identifier, task] = dict(
                        source_inp=str(inp), source_odb=str(inp.with_suffix(".odb")), transfer=str(transfer))
            census.append(dict(topology=topology, design=design, eligible_paired_count=len(eligible),
                               eligibility="Both UT/FT original INP/ODB and retained transfer present, unique archive source; original IDs1-1000 only"))
    selection = paired_gallery_selection(candidates, seed)
    records = []
    for item in selection:
        for task in ("UT", "FT"):
            row = dict(item, task=task, **sources[item["topology"], item["design"], item["id"], task])
            row["native"] = str(folder/("{}_{}_{}_{}.npz".format(item["topology"], task, item["design"], item["id"])))
            records.append(row)
    manifest = dict(seed=seed, selection_rule="Stratified random:4full+3mechanism paired architecture IDs per topology; no response used in selection",
                    census=census, selected_pairs=selection, periodic=periodic, records=records,
                    native_support="No padding, stretching or shared private endpoint axes", new_solver_jobs=0)
    path = folder/"selection.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to redraw a frozen paired-gallery selection")
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def sampled_odb_fingerprint(path):
    """Cheap identity/read-only guard; explicitly not a full-file checksum."""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(65536))
        stream.seek(max(0, stat.st_size-65536))
        digest.update(stream.read(65536))
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                first_and_last_64KiB_sha256=digest.hexdigest(), full_file_hash=False)


def extract_paired_gallery(root):
    """Abaqus native read-only histories for the frozen56 disordered task cases."""
    from odbAccess import openOdb
    folder = Path(root)/"audit/curve-processing/paired-examples"
    selection = json.loads((folder/"selection.json").read_text())
    index_path = folder/"native-index.json"
    prior = {row["native"]: row for row in json.loads(index_path.read_text())} if index_path.exists() else {}
    records = []
    for row in selection["records"]:
        inp, source, transfer, native = (Path(row[name]) for name in ("source_inp", "source_odb", "transfer", "native"))
        before, inp_sha, transfer_sha = sampled_odb_fingerprint(source), sha(inp), sha(transfer)
        old = prior.get(str(native))
        if old and old["source_odb_fingerprint"] == before and old["source_inp_sha256"] == inp_sha and old["transfer_sha256"] == transfer_sha and native.exists() and old["native_sha256"] == sha(native):
            records.append(old)
            continue
        odb = openOdb(str(source), readOnly=True)
        try:
            if len(odb.steps) != 1:
                raise ValueError("Expected one original nonlinear step")
            step = list(odb.steps.values())[0]
            if row["task"] == "FT":
                regions = [step.historyRegions["Node ASSEMBLY.1"]]
            else:
                nodes = [node for group in odb.rootAssembly.nodeSets["SET-TOP"].nodes for node in group]
                regions = []
                for node in nodes:
                    found = [region for name, region in step.historyRegions.items()
                             if name.startswith("Node PART-1-1.") and name.endswith("."+str(node.label))
                             and "U2" in region.historyOutputs and "RF2" in region.historyOutputs]
                    if len(found) != 1:
                        raise ValueError("Ambiguous native clamp history")
                    regions.append(found[0])
            time = np.asarray([t for t, value in regions[0].historyOutputs["U2"].data])
            displacement, reaction = [], []
            for region in regions:
                for variable, target in (("U2", displacement), ("RF2", reaction)):
                    values = np.asarray(region.historyOutputs[variable].data)
                    if not np.array_equal(values[:, 0], time):
                        raise ValueError("Native paired histories have different timestamps")
                    target.append(values[:, 1])
            raw_x, raw_y = np.mean(displacement, axis=0), np.sum(reaction, axis=0)
        finally:
            odb.close()
        if sampled_odb_fingerprint(source) != before:
            raise ValueError("Native source changed during read-only extraction")
        retained = np.loadtxt(str(transfer), delimiter=",", usecols=(0, 1))
        common, scales, errors = min(len(time)-1, len(retained)), [], []
        for raw, previous in ((raw_x, retained[:, 0]), (raw_y, retained[:, 1])):
            eligible = (abs(raw[:common]) > np.max(abs(raw[:common]))*1e-7) & (previous[:common] != 0)
            if not np.any(eligible):
                raise ValueError("No native/transfer normalisation overlap")
            factor = float(np.median(previous[:common][eligible]/raw[:common][eligible]))
            error = float(np.max(abs(previous[:common][eligible]-raw[:common][eligible]*factor))/np.max(abs(previous[:common][eligible])))
            if error > 2e-5:
                raise ValueError("Native/transfer normalisation mismatch: " + str(transfer))
            scales.append(factor); errors.append(error)
        x, y = raw_x*scales[0], raw_y*scales[1]
        if not all(np.isfinite(value).all() for value in (time, x, y)) or np.any(np.diff(time) <= 0):
            raise ValueError("Native curve must be finite and time-increasing")
        np.savez_compressed(str(native), time=time, x=x, y=y)
        result = dict(row, source_odb_fingerprint=before, source_inp_sha256=inp_sha,
                      transfer_sha256=transfer_sha, native_sha256=sha(native), samples=len(time),
                      source_unchanged=True, read_only=True, normalisation_factors=scales,
                      native_transfer_relative_errors=errors, observed_support_only=True)
        records.append(result)
        index_path.write_text(json.dumps(records, indent=2)+"\n")
        print(row["topology"], row["task"], row["design"], row["id"], len(time), flush=True)
    return records


def plot_paired_gallery(root):
    """Eight figures, each periodic+seven random paired architectures, no property edits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = Path(root)/"audit/curve-processing/paired-examples"
    selection = json.loads((folder/"selection.json").read_text())
    native = json.loads((folder/"native-index.json").read_text())
    registry_path = Path(root)/"audit/curve-processing/periodic-cutoffs.json"
    registry = json.loads(registry_path.read_text())
    controls = {(row["topology"], row["task"]): row for row in registry["families"]}
    result, figures = [], []
    for topology in TOPOLOGIES:
        pairs = [row for row in selection["selected_pairs"] if row["topology"] == topology]
        for task in ("UT", "FT"):
            current = [row for row in selection["periodic"] if row["topology"] == topology and row["task"] == task]
            current += [next(row for row in native if row["topology"] == topology and row["task"] == task and row["design"] == pair["design"] and row["id"] == pair["id"]) for pair in pairs]
            if len(current) != 8:
                raise ValueError("One periodic plus seven paired models required")
            control = controls[topology, task]
            if control["status"] != "accepted_physical_mode":
                raise ValueError("Periodic physical mode not accepted")
            fig, axes = plt.subplots(4, 2, figsize=(13, 13.8), squeeze=False)
            for axis, row in zip(axes.flat, current):
                if sha(row["native"]) != row["native_sha256"]:
                    raise ValueError("Native gallery extraction changed")
                with np.load(row["native"], allow_pickle=False) as data:
                    time, x, y = (data[key] for key in ("time", "x", "y"))
                scale = 1. if task == "UT" else .001
                axis.plot(x, y*scale, color=".65", lw=.85, label="Raw", zorder=1)
                axis.plot(x, np.asarray(smooth(smooth(y)))*scale, color="#6f5094", ls="--", lw=1.4, label="Legacy double moving average", zorder=3)
                nyquist = float((len(time)-1)/(2*(time[-1]-time[0])))
                if control["cutoff_Hz"] < nyquist:
                    (fx, fy), info = periodic_causal_filter(time, x, y, control["natural_frequency_Hz"])
                    axis.plot(fx, fy*scale, color=TOPOLOGY_COLORS[topology], lw=1.25, label="Causal Butterworth, periodic2fn", zorder=4)
                    status = "applied_causal_Butterworth_2fn"
                else:
                    status = "sampling_unresolved_cutoff_at_or_above_saved_Nyquist"
                    info = None
                design = row["design"]
                title = "Periodic" if design == "periodic" else ("Full disorder" if design == "full" else "Mechanism-based disorder")+", ID "+str(row["id"])
                axis.set_title(title, color={"periodic":".1", "full":"#ff7f0e", "mechanism":"#1f77b4"}[design], fontsize=10)
                note = r"$2f_n$="+format(control["cutoff_Hz"], ".1f")+"Hz; Nyquist="+format(nyquist, ".1f")+"Hz"
                if info is None:
                    note += "\nButterworth unavailable at saved sampling"
                axis.text(.98, .04, note, transform=axis.transAxes, ha="right", va="bottom", fontsize=8,
                          bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=2))
                axis.set_xlabel(r"Macroscopic strain, $\varepsilon_L$" if task == "UT" else "Load-line displacement (mm)", fontsize=9)
                axis.set_ylabel(r"Macroscopic stress, $\Sigma$ (MPa)" if task == "UT" else "Force (kN)", fontsize=9)
                axis.grid(alpha=.18)
                result.append(dict(topology=topology, task=task, design=design, id=row["id"],
                    native_samples=len(time), periodic_cutoff_Hz=control["cutoff_Hz"], saved_Nyquist_Hz=nyquist,
                    filter_status=status, filter_information=info, native=str(row["native"]), native_sha256=row["native_sha256"]))
            handles = [Line2D([], [], color=".65", label="Raw response"),
                       Line2D([], [], color=TOPOLOGY_COLORS[topology], label=r"Causal Butterworth, periodic $2f_n$ (where resolved)"),
                       Line2D([], [], color="#6f5094", ls="--", label="Legacy double moving average")]
            title = {"tri":"Triangular", "kagome":"Kagome", "hex":"Hexagonal", "FCC":"FCC"}[topology]
            fig.suptitle(title+" | "+task+" | paired architecture selection, seed "+str(selection["seed"]), y=.995, fontsize=13)
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .977), ncol=3, frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, .945), h_pad=1.8, w_pad=1.3)
            for extension in ("png", "svg"):
                target = folder/(topology+"-"+task.lower()+"-paired-curves."+extension)
                fig.savefig(target, dpi=190)
                figures.append(str(target))
            plt.close(fig)
    report = dict(seed=selection["seed"], selection_rule=selection["selection_rule"], figures=figures,
        paired_architectures=selection["selected_pairs"], histories=result, periodic_registry_sha256=sha(registry_path),
        legacy_method="Twice resources.calculations.smooth (default5-point moving average), force/stress only against original paired coordinates",
        accepted_filter="Paired causal order2/DC Butterworth at twice exact periodic specimen fn; no capped cutoff or invented high-rate data",
        original_simulations_and_properties_unchanged=True, new_solver_jobs=0)
    (folder/"gallery.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def first_crossing(y, fraction):
    peak = int(np.argmax(y))
    values = np.flatnonzero(np.asarray(y)[peak:] <= fraction*y[peak])
    return int(peak+values[0]) if len(values) else None


def threshold_details(x, y, fraction=.25):
    """Keep the recorded crossing distinct from a display-only interpolation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    peak = int(np.argmax(y))
    crossing = first_crossing(y, fraction)
    threshold = float(fraction*y[peak])
    result = {'peak_index': peak, 'peak_strain': float(x[peak]),
              'peak_MPa': float(y[peak]), 'threshold_MPa': threshold,
              'crossing_index': crossing, 'criterion': 'first recorded post-peak sample at or below 25% of that curve peak'}
    if crossing is None:
        return dict(result, recorded_ductility=None, interpolated_crossing_for_display=None)
    interpolated = float(x[crossing])
    if crossing > peak and y[crossing-1] != y[crossing]:
        weight = (threshold-y[crossing-1])/(y[crossing]-y[crossing-1])
        interpolated = float(x[crossing-1]+weight*(x[crossing]-x[crossing-1]))
    return dict(result, recorded_ductility=float(x[crossing]),
                interpolated_crossing_for_display=interpolated)


def extract_hex_ductility_native(root):
    """Abaqus Python: recover all126 genuine samples without solving/upgrading."""
    from odbAccess import openOdb
    root = Path(root)
    receipt = root/'audit/production/native-normalisation-replay-20261102.json'
    original = next(r for r in json.loads(receipt.read_text())['cases'] if r['lat']=='hex')
    path = Path(original['source_odb'])
    before = sha(path)
    if before != original['after_sha256']:
        raise ValueError('Native hex source differs from retained replay')
    odb = openOdb(str(path), readOnly=True)
    try:
        if len(odb.steps) != 1:
            raise ValueError('Expected one source step')
        step = list(odb.steps.values())[0]
        nodes = [node for group in odb.rootAssembly.nodeSets['SET-TOP'].nodes for node in group]
        histories = []
        for node in nodes:
            matches = [(name, region) for name, region in step.historyRegions.items()
                       if name.startswith('Node ') and name.endswith('.'+str(node.label))
                       and 'RF2' in region.historyOutputs and 'U2' in region.historyOutputs]
            if len(matches) != 1:
                raise ValueError('Ambiguous source top-node history')
            histories.append(matches[0])
        time = np.array([t for t, _ in histories[0][1].historyOutputs['RF2'].data])
        force, travel = [], []
        for _, region in histories:
            for name, values in (('RF2', force), ('U2', travel)):
                data = region.historyOutputs[name].data
                if not np.array_equal(time, np.array([t for t, _ in data])):
                    raise ValueError('Native top history grids differ')
                values.append([v for _, v in data])
        stress = np.sum(force, axis=0)/original['inferred_area_median_mm2']
        strain = np.mean(travel, axis=0)/original['inferred_height_median_mm']
    finally:
        odb.close()
    if before != sha(path) or len(time) != original['native_samples']:
        raise ValueError('Native source changed or sample count differs')
    if not all(np.isfinite(v).all() for v in (time, strain, stress)):
        raise ValueError('Finite genuine native prefix required')
    result = {'scope':'Hexagonal full-disorder ID1 UAT; native top-clamp histories, no new simulation',
              'source_odb':str(path),'source_odb_sha256':before,'source_unchanged':True,
              'normalisation_receipt':str(receipt),'normalisation_receipt_sha256':sha(receipt),
              'area_mm2':original['inferred_area_median_mm2'],
              'gauge_height_mm':original['inferred_height_median_mm'],
              'native_samples':len(time),'top_nodes':len(nodes),
              'time_s':time.tolist(),'strain':strain.tolist(),'stress_MPa':stress.tolist(),
              'recorded_time_step_range_s':[float(np.diff(time).min()),float(np.diff(time).max())],
              'source_input':original['source_input'],'source_input_sha256':original['source_input_sha256'],
              'read_only':True,'ODB_upgraded':False,'simulation_run':False}
    out = root/'audit/curve-processing/filtered-curves/hex-ductility-filtering.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def plot_hex_ductility(root):
    """Show the isolated raw threshold dip and a declared cutoff sensitivity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path = Path(root)/'audit/curve-processing/filtered-curves/hex-ductility-filtering.json'
    data = json.loads(path.read_text())
    if sha(data['source_odb']) != data['source_odb_sha256']:
        raise ValueError('Hex source changed since native extraction')
    x, y = np.array(data['strain']), np.array(data['stress_MPa'])
    raw = threshold_details(x,y)
    selected_x, selected_y = paired_filter(x,y,.4)
    selected = threshold_details(selected_x,selected_y)
    old_y = np.asarray(smooth(smooth(y)))
    old = threshold_details(x,old_y)
    corners = (.1,.2,.3,.4,.5,.6,.8)
    sensitivities = []
    for corner in corners:
        xx, yy = paired_filter(x,y,corner)
        metric = threshold_details(xx,yy)
        sensitivities.append(dict(metric, corner_fraction_Nyquist=corner,
            equivalent_cutoff_Hz_if_median_native_dt=corner/(2*np.median(np.diff(data['time_s']))),
            ductility_change_percent=100*(metric['recorded_ductility']/raw['recorded_ductility']-1)))
    color = TOPOLOGY_COLORS['hex']
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig, axes = plt.subplots(1,3,figsize=(13.2,3.8),gridspec_kw={'width_ratios':[1.12,1.12,1.]})
        for ax in axes[:2]:
            ax.plot(100*x,y,color='.55',lw=1.1,label='Raw')
            ax.plot(100*selected_x,selected_y,color=color,lw=1.6,label='Butterworth')
            ax.plot(100*x,old_y,color='#9467bd',lw=1.,ls=':',label='Previous moving average')
            for metric, style, col, xx, yy in ((raw,'--','.45',x,y),(selected,'-',color,selected_x,selected_y)):
                p, c = metric['peak_index'],metric['crossing_index']
                ax.plot(100*xx[p],yy[p],'o',ms=4,mfc='white',mec=col)
                ax.axhline(metric['threshold_MPa'],color=col,ls=style,lw=.9,alpha=.8)
                ax.plot(100*xx[c],yy[c],'o',ms=5,mfc='white',mec=col)
                ax.vlines(100*xx[c],0,yy[c],color=col,ls='--',lw=1.)
            ax.set_xlabel(r'Macroscopic strain, $\varepsilon_L$ (%)')
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(r'Macroscopic stress, $\Sigma$ (MPa)')
        axes[0].set_title('(a) Complete genuine history')
        axes[0].set_ylim(bottom=0)
        axes[0].legend(frameon=False,fontsize=8.5,loc='upper right')
        axes[1].set_title('(b) Ductility threshold')
        axes[1].set_xlim(6.4,8.2)
        axes[1].set_ylim(0,10.5)
        axes[1].text(.025,.95,'Raw: %.3f%%\nFiltered: %.3f%%\nChange: +%.2f%%' % (
            100*raw['recorded_ductility'],100*selected['recorded_ductility'],
            100*(selected['recorded_ductility']/raw['recorded_ductility']-1)),
            transform=axes[1].transAxes,va='top',fontsize=9,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.88})
        axes[2].plot(corners,[100*r['recorded_ductility'] for r in sensitivities],
                     'o-',color=color,lw=1.2,ms=4)
        axes[2].axhline(100*raw['recorded_ductility'],color='.45',ls='--',lw=1.,label='Raw')
        axes[2].axhline(100*old['recorded_ductility'],color='#9467bd',ls=':',lw=1.,label='Previous moving average')
        axes[2].plot(.4,100*selected['recorded_ductility'],'o',ms=7,mfc='white',mec=color,mew=1.3)
        axes[2].set_xlabel(r'Corner / sample Nyquist, $f_c/f_N$')
        axes[2].set_ylabel(r'Ductility, $\mathcal{D}$ (%)')
        axes[2].set_title('(c) Cutoff sensitivity')
        axes[2].grid(alpha=.15)
        axes[2].legend(frameon=False,fontsize=8.5)
        fig.suptitle('Hexagonal full disorder, ID 1',fontsize=12,y=.99)
        fig.subplots_adjust(top=.82,bottom=.17,left=.055,right=.99,wspace=.32)
        for extension in ('png','pdf','svg'):
            fig.savefig(path.with_suffix('.'+extension),dpi=220)
        plt.close(fig)
    data.update(raw=raw,butterworth_corner_0p4=selected,previous_moving_average=old,
                sensitivities=sensitivities,
                filter={'order':2,'implementation':'SOS forward/backward on native paired samples',
                        'selected_corner_fraction_Nyquist':.4,'corner_selection':'Gentlest of original pilot corners .1,.2,.4; not a modal or Hsieh calibration',
                        'original_pilot_corners':[.1,.2,.4],
                        'definition':'f_N=fs/2, so normalized .4 means fc=.2fs',
                        'median_native_dt_s':float(np.median(np.diff(data['time_s']))),
                        'equivalent_fc_Hz':float(.4/(2*np.median(np.diff(data['time_s'])))),
                        'equivalent_Hz_interpretation':'Native output times are nearly uniform; this Hz value is explanatory, not an eigenfrequency-qualified cutoff.',
                        'paired_coordinate_note':'Neither time nor strain range is stretched. The existing pilot filters both continuous paired channels; x-filter displacement near crossings is recorded separately.'},
                prior_125_sample_transfer_comparison={'raw_ductility':.06927096247673034,
                    'filtered_ductility':.0767079827406632,'change_percent':10.73612953830565,
                    'reason':'Prior figure used125 genuine common CSV samples; this closeup recovers all126 native samples.'},
                caption='Hexagonal full-disorder ID1 native UAT history. Grey raw and green second-order Butterworth at0.4 sample Nyquist are compared with the previous moving average (purple dotted). Hollow circles mark peaks and first recorded post-peak25%crossings; horizontal lines are25%of each respective peak, vertical guides mark the corresponding recorded ductility strain. The raw early crossing is an isolated dip; filtering delays the crossing, and the old moving average gives essentially the same later result. Panel(c) shows a predeclared broad cutoff sensitivity, not a cutoff fitted to a target ductility. Lines join recorded samples; interpolated crossing coordinates are retained separately for display/diagnosis and do not replace the recorded criterion. This is a filter sensitivity, not proof that either raw or filtered threshold is a physically validated target.')
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    return data


def query_support_mask(observed_x, query_x):
    """Observation support on a declared physical query axis, not frame count.

    This helper accepts one verified contiguous prefix only. It does not bridge
    missing intervals, infer a failed specimen's tail, or manufacture STATUS.
    """
    observed_x, query_x = np.asarray(observed_x, float), np.asarray(query_x, float)
    if (len(observed_x) < 2 or not np.isfinite(observed_x).all()
            or not np.isfinite(query_x).all() or np.any(np.diff(observed_x) < 0)):
        raise ValueError('One finite nondecreasing observed support interval is required')
    return (query_x >= observed_x[0]) & (query_x <= observed_x[-1])


def event_segmented_filter(x, y, split, corner):
    """Test filtering each side of a recorded deletion independently.

    No endpoint is pinned and no artificial sample is introduced. Short sides
    remain raw and are explicitly unresolved; this is a sensitivity control.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if split is None or not 0 < split < len(x):
        return None, {'status': 'no_recorded_partial_STATUS_split'}
    pieces, qualified = [], []
    for start, stop in ((0, int(split)), (int(split), len(x))):
        xx, yy = x[start:stop], y[start:stop]
        if len(xx) > 12:
            xx, yy = paired_filter(xx, yy, corner)
            qualified.append(True)
        else:
            qualified.append(False)
        pieces.append((xx, yy))
    return tuple(np.concatenate([p[i] for p in pieces]) for i in (0, 1)), {
        'status': 'tested_both_sides' if all(qualified) else 'short_side_kept_raw_unresolved',
        'first_post_deletion_index': int(split), 'filtered_sides': qualified,
        'event_basis': 'Transfer partial STATUS loss; native local initiation is not attested here.',
        'endpoint_pinning': False}


def numeric_metrics(x, y, fixed_end):
    crossing = first_crossing(y, .25)
    span = float(np.ptp(y))
    return {"peak": float(max(y)),
            "work_fixed_raw_endpoint": float(np.trapz(y[:fixed_end+1], x[:fixed_end+1])),
            "end_load_fixed_raw_endpoint": float(y[fixed_end]),
            "ductility_25": float(x[crossing]) if crossing is not None else None,
            "ductility_25_sample": crossing,
            "second_difference_RMS_over_span": float(np.sqrt(np.mean(np.diff(y, n=2)**2))/span) if span else None}


def transfer_prefix(path, known_n=None):
    """Read continuous columns; keep transfer support distinct from native proof."""
    points, first_deletion, first_all_zero_status = [], None, None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.reader(stream)):
            points.append([float(row[0]), float(row[1])])
            if len(row) > 2:
                status = np.asarray([float(v) for v in row[2:]], float)
                if first_all_zero_status is None and len(status) and np.all(status == 0):
                    first_all_zero_status = index
                if first_deletion is None and np.any(status == 0) and not np.all(status == 0):
                    first_deletion = index
    points = np.asarray(points)
    stop = len(points) if known_n is None else int(known_n)
    reason = "native_prefix_bound_from_retained_replay" if known_n is not None else "transfer_only_native_support_not_attested"
    if first_all_zero_status is not None:
        stop = min(stop, first_all_zero_status)
    # A zero-valued transfer tail is never accepted as observed structural failure.
    tail = np.flatnonzero(np.any(points != 0., axis=1))
    if len(tail):
        stop = min(stop, int(tail[-1]+1))
    finite = np.flatnonzero(~np.isfinite(points[:stop]).all(axis=1))
    if len(finite):
        stop = int(finite[0])
    return points[:stop], {"CSV_rows": len(points), "used_prefix_rows": stop,
                           "native_support": reason, "first_partial_STATUS_deletion_row": first_deletion,
                           "first_all_zero_STATUS_row": first_all_zero_status,
                           "no_tail_imputed": True}


def native_filter_controls(root):
    """Use only existing source-matched modal controls and fixed material events."""
    from validation_calculations import filter_force
    modal = Path(root)/"audit/modal/summary.json"
    records = []
    for control in json.loads(modal.read_text()):
        source = Path(control["source_odb"])
        relative = source.relative_to(root)
        event_path = Path(root)/"audit/failure"/relative.parent/source.stem/"initiation.json"
        curve_path = Path(root)/"audit/history"/relative.parent/source.stem/"native_curve.csv"
        if not event_path.is_file() or not curve_path.is_file():
            continue
        event = json.loads(event_path.read_text())
        if not event.get("prev_native_history"):
            continue
        with curve_path.open(newline="") as stream:
            curve = list(csv.DictReader(stream))
        time = np.array([float(row["time_s"]) for row in curve])
        x = np.array([float(row["U2_mm"]) for row in curve])
        y = np.array([float(row["RF2_N"]) for row in curve])
        endpoint = int(event["prev_native_history"]["index"])
        raw_work = float(np.trapz(y[:endpoint+1], x[:endpoint+1]))
        for factor in (.5, 1., 2.):
            cutoff = control["cutoff_2fn_Hz"]*factor
            for support in ("through_fixed_material_endpoint_only", "whole_native_history"):
                stop = endpoint+1 if support.startswith("through") else len(time)
                row = {"case_id": control["case_id"], "modal_control": control["control"],
                       "cutoff_Hz": cutoff, "cutoff_multiplier": factor,
                       "filter_support": support, "fixed_endpoint_index": endpoint,
                       "fixed_endpoint_source": str(event_path), "event_source_sha256": sha(event_path),
                       "curve_source": str(curve_path), "curve_source_sha256": sha(curve_path),
                       "modal_source_sha256": sha(modal)}
                try:
                    xx, info = filter_force(time[:stop], x[:stop], cutoff, zero_phase=True)
                    yy, _ = filter_force(time[:stop], y[:stop], cutoff, zero_phase=True)
                    work = float(np.trapz(yy[:endpoint+1], xx[:endpoint+1]))
                    row.update(status="evaluated", filter_info=info,
                               work_relative_change=work/raw_work-1,
                               fixed_endpoint_force_relative_change=float(yy[endpoint]/y[endpoint]-1),
                               filtered_endpoint_force_N=float(yy[endpoint]),
                               raw_endpoint_force_N=float(y[endpoint]))
                except ValueError as error:
                    row.update(status="rejected_sampling_or_support", reason=str(error))
                records.append(row)
    return records


def run(root, data_root, out):
    root, data_root, out = Path(root), Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    replay = json.loads((root/"audit/production/native-normalisation-replay-20261102.json").read_text())
    known = {row["raw_OUT"]: row["compared_genuine_prefix_samples"] for row in replay["cases"]}
    specifications = []
    for topology, nx in TOPOLOGIES.items():
        for mode in ("Ductile", "Fracture"):
            specifications.append((topology, "periodic", 0, mode,
                                   data_root/"per"/topology/"transfer"/f"OUT-{mode}-{topology}-{nx}-per-1.csv"))
            for design, part, token in (("full", "", "all"), ("mechanism", "Target-xs", "xs")):
                for identity in IDS:
                    specifications.append((topology, design, identity, mode,
                        data_root/"disNodes"/part/"0.2"/topology/"transfer"/
                        f"OUT-{mode}-{topology}-{nx}-20disNodes-lhs-{token}-{identity}.csv"))
    cases, filters, sources = [], [], {}
    for topology, design, identity, mode, path in specifications:
        if not path.is_file():
            cases.append({"topology": topology, "design": design, "id": identity, "mode": mode,
                          "path": str(path), "status": "missing_source"})
            continue
        before = sha(path)
        sources[str(path)] = before
        points, support = transfer_prefix(path, known.get(str(path)))
        x, y = points.T
        if len(x) < 13:
            cases.append({"topology": topology, "design": design, "id": identity, "mode": mode,
                          "path": str(path), "support": support, "status": "insufficient_prefix"})
            continue
        if mode == "Fracture" and support["first_partial_STATUS_deletion_row"] is not None:
            endpoint = max(1, min(len(x)-1, support["first_partial_STATUS_deletion_row"]-1))
            endpoint_role = "last_transfer_sample_before_partial_STATUS_loss_not_native_initiation"
        else:
            crossing = first_crossing(y, .25)
            endpoint = crossing if crossing is not None else len(x)-1
            endpoint_role = "fixed_raw_25pct_crossing_or_last_observed_prefix_not_material_initiation"
        raw = numeric_metrics(x, y, endpoint)
        record = {"topology": topology, "design": design, "id": identity, "mode": mode,
                  "path": str(path), "source_sha256": before, "support": support,
                  "fixed_endpoint_index": endpoint, "endpoint_role": endpoint_role,
                  "raw_metrics": raw, "status": "analysed"}
        if mode == "Ductile":
            record["initial_fit"] = initial_fit(x[:20], y[:20])
            if topology == "hex":
                record["expanded_strain_cap_sensitivity"] = {
                    str(cap): initial_fit(x[:30], y[:30], max_strain=cap)
                    for cap in (.01, .02, .03)}
        cases.append(record)
        methods = [("double_moving_average", x, np.asarray(smooth(smooth(y))), "whole_observed_prefix")]
        for corner in (.1, .2, .4):
            xx, yy = paired_filter(x, y, corner)
            name = f"butterworth2_zero_phase_corner_{corner}_Nyquist"
            methods.append((name, xx, yy, "whole_observed_prefix"))
            if endpoint >= 12:
                xx, yy = paired_filter(x[:endpoint+1], y[:endpoint+1], corner)
                methods.append((name, xx, yy, "through_fixed_endpoint_only"))
        for name, xx, yy, scope in methods:
            values = numeric_metrics(xx, yy, endpoint)
            baseline = raw if scope == "whole_observed_prefix" else numeric_metrics(x[:endpoint+1], y[:endpoint+1], endpoint)
            work_change = (values["work_fixed_raw_endpoint"]/baseline["work_fixed_raw_endpoint"]-1) if baseline["work_fixed_raw_endpoint"] else None
            peak_change = values["peak"]/baseline["peak"]-1 if baseline["peak"] else None
            duct_change = (values["ductility_25"]/baseline["ductility_25"]-1) if values["ductility_25"] is not None and baseline["ductility_25"] else None
            load_change = values["end_load_fixed_raw_endpoint"]/raw["end_load_fixed_raw_endpoint"]-1 if raw["end_load_fixed_raw_endpoint"] else None
            # Numerical screening thresholds declared a priori; not proof of physics.
            passes = bool(work_change is not None and abs(work_change) <= .01 and abs(peak_change) <= .02
                          and (mode != "Ductile" or (duct_change is not None and abs(duct_change) <= .02)))
            filters.append({"topology": topology, "design": design, "id": identity, "mode": mode,
                            "method": name, "filter_support": scope, "work_relative_change": work_change,
                            "peak_relative_change": peak_change, "ductility_relative_change": duct_change,
                            "fixed_endpoint_load_relative_change": load_change,
                            "numerical_preservation_screen_passes": passes, "metrics": values,
                            "x_monotone": bool(np.all(np.diff(xx) >= 0)),
                            "initial_fit_slope_change": None})
            if mode == "Ductile" and record["initial_fit"].get("selected"):
                end = record["initial_fit"]["selected"]["end_index"]
                filtered_slope = fit_line(xx[:end+1], yy[:end+1])[0]
                native_slope = fit_line(x[:end+1], y[:end+1])[0]
                filters[-1]["initial_fit_slope_change"] = filtered_slope/native_slope-1
    assert all(sha(Path(path)) == digest for path, digest in sources.items()), "Source changed during pilot"
    result = {"status": "Completed source-preserving processing pilot; production route unchanged",
              "scope": "Four topologies, periodic plus full/mechanism original IDs 1,50,500,1000; UAT and FT",
              "source_sha256": sources, "sources_unchanged": True, "cases": cases, "filters": filters,
              "native_fixed_material_event_filter_controls": native_filter_controls(root),
              "filter_rule": {"order": 2, "application": "forward-backward SOS, paired continuous coordinates/response",
                              "corners_fraction_Nyquist": [.1,.2,.4], "physical_Hz_cutoff_qualified": False,
                              "screen": "|work change|<=1%, |peak change|<=2%, |UAT ductility change|<=2% at fixed raw endpoint",
                              "screen_interpretation": "Sensitivity flag only. Raw noisy maxima are not ground truth, so failing raw-peak preservation does not itself disqualify denoising.",
                              "limitations": "Sample positions are not recorded time. Zero-phase endpoint/future-information and aliasing remain checks; no STATUS or missing tails filtered."},
              "fit_rule": {"minimum_samples_after_startup": 6, "starts": [0,1,2], "end_indices": [7,19],
                           "maximum_strain_span": .005, "span_normalised_RMS_limit": .02,
                           "startup_and_neighbour_slope_limit": .02, "consecutive_windows": 3,
                           "selection": "Central member of earliest stable initial family; no averaging invalid windows",
                           "material_preplasticity_and_body_gauge_verified": False},
              "mask_migration": ["Keep native timestamps and observation masks; do not zero-fill missing fracture observations.",
                  "Fit scalers only to observed training-fold values. Store finite placeholders plus explicit masks for tensor operations.",
                  "Apply masks before loss arithmetic, including derivatives, work integrals and metrics; require contiguous support for differences.",
                  "Do not give unknown test-time termination masks to the network as an input: that leaks a response-derived failure endpoint.",
                  "Compare identical architecture/folds/seeds on raw+mask, legacy smoothing+mask and qualified Butterworth+mask; maintain existing winner until holdout evidence.",
                  "Evaluate observed-support curve errors, peak/ductility/work bias, recovered-property errors and results versus observed coverage; no arbitrary training of missing tails as zero.",
                  "Retain independently observed initiation/property labels; filtering a training target must not redefine material onset."],
              "limits": ["Transfer-only support for most cases is not a native ODB attestation.",
                         "Numerically stable grip-based slopes are candidates, not accepted elastic material moduli.",
                         "No fitted cutoff is chosen to improve reference agreement; no source dataset or trained model is changed."]}
    (out/"curve-processing.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def plot_examples(root, corner=.4):
    """Actual periodic/disordered curves; the sample filter is not a modal claim."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root=Path(root)
    pilot_path=root/'audit/curve-processing/curve-processing.json'
    pilot=json.loads(pilot_path.read_text())
    out=pilot_path.parent/'filtered-curves'
    out.mkdir(exist_ok=True)
    chosen=[r for r in pilot['cases'] if r.get('status')=='analysed'
            and ((r['design']=='periodic' and r['id']==0)
                 or (r['design'] in ('full','mechanism') and r['id']==1))]
    if len(chosen)!=24:
        raise ValueError('Expected UAT/FT examples for four topologies and three designs')
    names={'tri':'Triangular','kagome':'Kagome','hex':'Hexagonal','FCC':'FCC'}
    designs=(('periodic','Periodic'),('full','Full disorder, ID 1'),
             ('mechanism','Mechanism-based disorder, ID 1'))
    points,metrics,sources=[],[],{}
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'}):
        for mode,stem in (('Ductile','uat-filtered-curves'),('Fracture','ft-filtered-curves')):
            fig,axes=plt.subplots(4,3,figsize=(12.3,11.7))
            for row,topology in enumerate(TOPOLOGIES):
                for column,(design,title) in enumerate(designs):
                    record=next(r for r in chosen if (r['topology'],r['design'],r['mode'])==(topology,design,mode))
                    source=Path(record['path'])
                    before=sha(source)
                    if before!=record['source_sha256']:
                        raise ValueError('Curve source changed since the processing pilot: '+str(source))
                    sources[str(source)]=before
                    known=(record['support']['used_prefix_rows'] if record['support']['native_support'].startswith('native_prefix') else None)
                    native,support=transfer_prefix(source,known)
                    x,y=native.T
                    xx,yy=paired_filter(x,y,corner)
                    moving=np.asarray(smooth(smooth(y)))
                    fixed_end=record['fixed_endpoint_index']
                    raw=numeric_metrics(x,y,fixed_end)
                    filtered=numeric_metrics(xx,yy,fixed_end)
                    old=numeric_metrics(x,moving,fixed_end)
                    def changes(values):
                        return {'peak_change_percent':100*(values['peak']/raw['peak']-1),
                                'paired_work_change_percent':100*(values['work_fixed_raw_endpoint']/raw['work_fixed_raw_endpoint']-1),
                                'fixed_endpoint_force_change_percent':100*(values['end_load_fixed_raw_endpoint']/raw['end_load_fixed_raw_endpoint']-1) if raw['end_load_fixed_raw_endpoint'] else None,
                                'ductility_change_percent':100*(values['ductility_25']/raw['ductility_25']-1) if mode=='Ductile' and values['ductility_25'] is not None and raw['ductility_25'] else None}
                    metric={'mode':mode,'topology':topology,'design':design,'id':record['id'],
                            'source':str(source),'source_sha256':before,'support':support,
                            'fixed_endpoint_index':fixed_end,'endpoint_role':record['endpoint_role'],
                            'raw':raw,'butterworth':filtered,'moving_average':old,
                            'butterworth_change':changes(filtered),'moving_average_change':changes(old)}
                    segmented, segmented_support = event_segmented_filter(
                        x, y, support['first_partial_STATUS_deletion_row'], corner)
                    metric['event_segmented_control'] = segmented_support
                    if segmented is not None:
                        segmented_values = numeric_metrics(*segmented, fixed_end)
                        metric['event_segmented_control'].update(
                            metrics=segmented_values, change=changes(segmented_values))
                    metrics.append(metric)
                    for index in range(len(x)):
                        points.append({'mode':mode,'topology':topology,'design':design,'id':record['id'],
                                       'sample':index,'raw_x':float(x[index]),'raw_y':float(y[index]),
                                       'butterworth_x':float(xx[index]),'butterworth_y':float(yy[index]),
                                       'moving_average_y':float(moving[index]),
                                       'source_sha256':before,'support':support['native_support']})
                    ax=axes[row,column]
                    xscale,yscale=(100.,1.) if mode=='Ductile' else (1.,.001)
                    ax.plot(x*xscale,y*yscale,color='.62',lw=.8,alpha=.9,zorder=1)
                    ax.plot(x*xscale,moving*yscale,color='#9467bd',lw=1.,ls='--',alpha=.85,zorder=2)
                    ax.plot(xx*xscale,yy*yscale,color=TOPOLOGY_COLORS[topology],lw=1.25,zorder=3)
                    if row==0:ax.set_title(title,pad=10)
                    if column==0:
                        ax.text(.025,.95,names[topology],transform=ax.transAxes,va='top',fontweight='bold',
                                bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
                    ax.grid(alpha=.15)
                    ax.set_axisbelow(True)
                    ax.tick_params(labelsize=8)
                    if row==3:
                        ax.set_xlabel(r'Macroscopic strain, $\varepsilon_L$ (%)' if mode=='Ductile' else 'Opening displacement (mm)')
                    if column==0:
                        ax.set_ylabel(r'Macroscopic stress, $\Sigma$ (MPa)' if mode=='Ductile' else 'Reaction force (kN)')
            handles=[Line2D([],[],color='.62',lw=1.,label='Raw response'),
                     Line2D([],[],color='#111111',lw=1.4,label='Butterworth (topology colour)'),
                     Line2D([],[],color='#9467bd',lw=1.,ls='--',label='Previous moving average')]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False)
            fig.suptitle(('UAT' if mode=='Ductile' else 'C(T)')+' responses: raw and filtered',y=.985,fontsize=13)
            fig.text(.5,.95,r'Second-order SOS, forward/backward; $f_c=0.4f_{\mathrm{Nyquist}}$ in sample space',ha='center',fontsize=9)
            fig.subplots_adjust(top=.91,bottom=.075,left=.085,right=.985,hspace=.29,wspace=.23)
            for extension in ('png','pdf','svg'):
                fig.savefig(out/(stem+'.'+extension),dpi=220)
            plt.close(fig)
    if any(sha(path)!=value for path,value in sources.items()):
        raise ValueError('A source changed during figure generation')
    with (out/'points.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    modal=json.loads((root/'audit/modal/summary.json').read_text())
    replay_path=root/'audit/production/native-normalisation-replay-20261102.json'
    replay=json.loads(replay_path.read_text())
    hex_replay=next(r for r in replay['cases'] if r['lat']=='hex')
    native_hex_end=hex_replay['mean_top_U2_last_mm']/hex_replay['inferred_height_median_mm']
    periodic_hex=next(r for r in chosen if (r['topology'],r['design'],r['mode'])==('hex','periodic','Ductile'))
    query_example=np.loadtxt(periodic_hex['path'],delimiter=',',usecols=(0,))
    # Audit the current common periodic axis; never stretch a short prefix.
    hex_mask=query_support_mask([0.,native_hex_end],query_example)
    comparison=[]
    for topology,nx in TOPOLOGIES.items():
        controls=[r for r in modal if r['job'].startswith('modal_'+topology+'_')]
        comparison.append({'topology':topology,'example_nx':nx,
                           'available_FT_modal_jobs':[r['job'] for r in controls],
                           'source_matched_modal_transfer_to_examples':False,
                           'reason':'Existing FT controls have different specimen sizes; no UAT or disorder modal transfer is established.' if controls else 'No corresponding retained modal control was found.'})
    metadata={'source_sha256':sources,'all_sources_unchanged':True,'source_pilot_sha256':sha(pilot_path),
              'filter':{'order':2,'corner_fraction_sample_Nyquist':corner,'implementation':'SOS forward/backward, paired continuous x/y',
                        'physical_Hz_frequency_assigned':False,'origin':'Sample-domain denoising sensitivity; not a periodic natural-frequency prescription.'},
              'modal_transfer_check':comparison,'metrics':metrics,
              'figure_conventions':{'strain_symbol':'varepsilon_L','topology_colours':TOPOLOGY_COLORS,
                                    'colour_authority':'Author correction and existing ValConvPlots.ipynb convergence curves.',
                                    'raw_colour':'grey','previous_moving_average_colour':'purple',
                                    'disorder_design':'Explicit column headings; curve colour identifies topology, not disorder design.'},
              'axes':{'Ductile':'Engineering strain from grip displacement/body height; stress MPa.',
                      'Fracture':'Opening mm and reaction force kN, original transfer values scaled only for plotting.'},
              'support':'Four full-ID1 UAT prefixes have retained native replay bounds. Other examples are transfer-only support, with artificial all-zero tails excluded; no absent observations invented.',
              'property_interpretation':'Peak/work/ductility changes are numerical sensitivities at a fixed raw endpoint; these examples do not independently certify material onset or a physical cutoff.',
              'mask_contract':{'axis_grid':'201 common reference positions per task/configuration; freeze shared planned-time/loading-progress meaning before migration, never specimen-wise rescaling',
                               'targets':'One finite filtered continuous-response value per observed query location.',
                               'mask':'Boolean [specimen,query]; true only within contiguous genuine native support. No STATUS interpolation.',
                               'placeholder':'Zero in normalised target space for masked positions; never a physical zero-force observation.',
                               'loss':'Mean over specimens of sum(mask*squared_error)/sum(mask), selecting valid entries before arithmetic.',
                               'derivative_mask':'Both adjacent query masks must be true; never cross a missing interval.',
                               'scalers':'Fit only observed entries in training folds.',
                               'property_masks':'Separate validity for onset, strength,25%-ductility and1%-work; uncrossed criteria remain censored.',
                               'network_inputs':'Existing geometry-only prediction can retain y-only outputs when all201 positions share one declared common axis. Unknown termination masks are not model inputs.'},
              'actual_support_example':{
                  'case':'Hexagonal full-disorder UAT ID 1',
                  'native_receipt':str(replay_path),'receipt_sha256':sha(replay_path),
                  'native_samples':hex_replay['native_samples'],
                  'retained_transfer_samples':hex_replay['compared_genuine_prefix_samples'],
                  'native_last_strain':native_hex_end,
                  'native_last_time_s':hex_replay['last_time_s'],
                  'reference_query_definition':'Actual201periodic strain values currently stored once bycreate_outputCSV; an audit illustration, not the selected final ML axis.',
                  'reference_query_source':periodic_hex['path'],
                  'reference_query':query_example.tolist(),
                  'support_mask':hex_mask.tolist(),'observed_query_count':int(hex_mask.sum()),
                  'why_frame_count_is_wrong':'126genuine time samples support124queries on the existing periodic strain reference. Time-frame count and actual-strain support are different measures. No126-to201stretch is proposed.',
                  'legacy_grid':'create_outputCSV stores the periodic strain row once, then the periodic stress row and stress-only disordered rows, each with a leading endpoint index. The model y-only output therefore relies on a shared reference-index meaning.',
                  'recommended_migration':'Preserve201common planned-time/loading-progress positions and mask unavailable positions; keep actual native strain/displacement paired for property calculations. Freeze this axis contract before ML changes.',
                  'exporter_limit':'The retained CSV omits the final genuine native sample. Plotted prefix has125 samples; recover the native126th sample before production migration.'},
              'property_filter_verdict':'Whole-history denoising is shown for visibility. It is not qualified as an automatic replacement for peak,25%-ductility or initiation force. Event-segmented controls are separate measured sensitivities; short segments remain unresolved. Native paired work and independently observed events remain authoritative pending a frozen endpoint rule.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    (out/'captions.txt').write_text(
        'Actual titanium periodic and disordered responses at nominal10mm/20%, with the four supervisor-selected specimen sizes. '
        'Raw is grey, the tested second-order Butterworth uses the topology colour (triangular red, kagome black, hexagonal green, FCC blue), and the previous repeated moving average is dashed purple. '
        'The Butterworth corner is0.4 of sample Nyquist; these panels do not use a verified periodic modal frequency. '
        'Forward/backward filtering is applied to the retained continuous paired coordinates/response. '
        'Unobserved all-zero transfer tails are excluded, not treated as measured zero response. '
        'Source support and individual peak/work/ductility sensitivities are recorded in metadata.json. '
        'Only the four full-disorder ID1 UAT prefixes have native replay attestation here; the remaining plotted transfer support is explicitly unverified. '
        'The plots demonstrate denoising and preserve raw data; they do not establish that every raw spike is physical or that one modal cutoff transfers to disorder.\n',encoding='utf-8')
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--data-root", type=Path, default=Path("Z:/p1/data/Ti"))
    parser.add_argument("--examples",action="store_true",help="Plot the actual periodic/full/mechanism examples without rerunning the pilot")
    parser.add_argument("--extract-hex-ductility",action="store_true",help="Abaqus Python: recover the complete native hex-ID1 history without solving")
    parser.add_argument("--hex-ductility",action="store_true",help="Plot native raw/filter thresholds and cutoff sensitivity")
    parser.add_argument("--periodic-cutoffs",action="store_true",help="Build authoritative periodic-only cutoff registry")
    parser.add_argument("--accept-physical-modes",action="store_true",help="After visual mode review, accept screened periodic modes")
    parser.add_argument("--extract-periodic-examples",action="store_true",help="Abaqus read-only extraction of24 genuine periodic/full/mechanism histories")
    parser.add_argument("--archive-root",type=Path,default=Path("Z:/p1/HPCarchive/Ti/data"))
    parser.add_argument("--apply-periodic-examples",action="store_true",help="Apply physical periodic cutoff to24 native histories; write two figures and metrics")
    parser.add_argument("--prepare-paired-examples",action="store_true",help="Freeze response-independent selection of7 paired disordered IDs per topology")
    parser.add_argument("--extract-paired-examples",action="store_true",help="Abaqus Python read-only extraction of56 selected native disordered histories")
    parser.add_argument("--paired-examples",action="store_true",help="Plot8paired UT/FT galleries, periodic plus7random architectures per topology")
    parser.add_argument("--selection-seed",type=int,default=20261110)
    args = parser.parse_args()
    if args.prepare_paired_examples:
        result=prepare_paired_gallery(args.root,args.archive_root,args.data_root,args.selection_seed)
        print(json.dumps(result["selected_pairs"],indent=2))
    elif args.extract_paired_examples:
        result=extract_paired_gallery(args.root)
        print('Recovered',len(result),'genuine native histories without solver submission')
    elif args.paired_examples:
        result=plot_paired_gallery(args.root)
        print('Wrote',len(result['figures']),'figure files from',len(result['histories']),'native histories')
    elif args.apply_periodic_examples:
        result=apply_periodic_examples(args.root)
        print('Wrote',len(result['figures']),'figure files and',len(result['measurements']),'method metrics')
    elif args.periodic_cutoffs:
        result=periodic_cutoff_registry(args.root,args.accept_physical_modes)
        print('Periodic family registry',len(result['families']))
    elif args.extract_periodic_examples:
        result=extract_periodic_cutoff_examples(args.root,args.archive_root,args.data_root)
        print('Recovered',len(result),'native histories without submitting simulations')
    elif args.extract_hex_ductility:
        result=extract_hex_ductility_native(args.root)
        print('Recovered',result['native_samples'],'unchanged native samples; no simulation run')
    elif args.hex_ductility:
        result=plot_hex_ductility(args.root)
        print('Wrote native hex-ID1 threshold comparison; raw/filtered ductility:',
              result['raw']['recorded_ductility'],result['butterworth_corner_0p4']['recorded_ductility'])
    elif args.examples:
        result=plot_examples(args.root)
        print('Wrote two actual-curve comparison figures from',len(result['source_sha256']),'unchanged sources')
    else:
        result = run(args.root, args.data_root, args.root/"audit/curve-processing")
        print("Completed", len(result["source_sha256"]), "source reads and", len(result["filters"]), "filter comparisons")

"""Read-only sampling-aware Butterworth and initial-work stiffness qualification.

The archive filter is an explicitly proposed adaptation, not a reconstruction
of missing solver-rate data. It never changes a source, MLdata or paper figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from resources.calculations import smooth
from validation_calculations import filter_force


ROOT = Path("Z:/p1/sims/Ti/Validation")
COLORS = {"tri": "#ff0000", "kagome": "#000000", "hex": "#008000", "FCC": "#0000ff"}
SOURCE = "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEOUTRefMap/simaout-c-dboutput-filter.htm"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_filter(time, x, y, natural_frequency_Hz, nyquist_fraction=1/3,
                   respect_resolved_physical=True):
    """Proposed causal order2/DC filtering, physical2fn or sampling fallback.

    The optional fraction is for declared sensitivity checks only. Abaqus' own
    automatic antialiasing uses fs/6 before output; applying that cutoff here is
    postprocessing and cannot reverse aliasing already present in the archive.
    """
    t, x, y = (np.asarray(v, dtype=float) for v in (time, x, y))
    if len(t) < 6 or len(t) != len(x) or len(t) != len(y):
        raise ValueError("At least six aligned native observations required")
    if not all(np.isfinite(v).all() for v in (t, x, y)) or np.any(np.diff(t) <= 0):
        raise ValueError("Finite, strictly increasing native times required")
    if not 0 < nyquist_fraction < 1 or not np.isfinite(natural_frequency_Hz) or natural_frequency_Hz <= 0:
        raise ValueError("Positive periodic frequency and a resolved sampling fraction required")
    nyquist = (len(t)-1)/(2*(t[-1]-t[0]))
    physical = 2*float(natural_frequency_Hz)
    cutoff = (physical if respect_resolved_physical and physical < nyquist else
              min(physical, float(nyquist_fraction)*nyquist))
    fx, info = filter_force(t, x, cutoff, order=2, zero_phase=False)
    fy, _ = filter_force(t, y, cutoff, order=2, zero_phase=False)
    info.update(policy="proposed_physical2fn_where_resolved_else_sampling_archive_adaptation", physical_cutoff_Hz=physical,
                sampling_cap_Hz=float(nyquist_fraction)*nyquist, capped=bool(2*natural_frequency_Hz > cutoff),
                default_cap="Unresolved physical cutoff only: fs/6; otherwise retain physical2fn", order=2, zero_phase=False,
                source_sampling_identity="native observations/timestamps retained; no new samples or tails",
                limitation="Postprocessing cannot undo existing aliasing or recreate original solver-rate filtering",
                source_basis=SOURCE)
    return (fx, fy), info


def initial_work_stiffness(x, y, end_fraction=.4, min_samples=6, baseline_response=0.):
    """Initial work-equivalent slope over one declared low-load interval.

    Match initial work W=E*x_end^2/2 through the first end_fraction crossing.
    Under linear elasticity this equals the ordinary modulus/load stiffness.
    The physical unloaded origin is used, not one noisy initial reaction.
    A known preload can be supplied explicitly as baseline_response.
    The estimate is not asserted to be a homogenised material tensor.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite observations required")
    if not 0 < end_fraction < 1:
        raise ValueError("Initial loading fraction must lie inside zero and one")
    if not np.isfinite(baseline_response):
        raise ValueError("Known finite baseline required")
    u, p = x-x[0], y-float(baseline_response)
    p=p.copy();p[0]=0.
    peak = int(np.argmax(p))
    if peak == 0 or p[peak] <= 0:
        return {"status": "unresolved_positive_initial_loading", "estimate": None}
    crossed = np.flatnonzero(p[:peak+1] >= end_fraction*p[peak])
    if not len(crossed):
        return {"status": "unobserved_initial_fraction", "estimate": None}
    end = int(crossed[0])
    a, b = u[:end+1], p[:end+1]
    if len(a) < 2 or np.any(np.diff(a) <= 0) or a[-1] <= 0:
        return {"status": "unresolved_monotone_initial_coordinate", "estimate": None, "end_index": end}
    work = np.r_[0., np.cumsum(.5*(b[1:]+b[:-1])*np.diff(a))]
    q = a*a
    estimate = float(2*work[-1]/q[-1])
    span = float(np.ptp(b))
    residual = float(np.sqrt(np.mean((b-estimate*a)**2))/max(span, 1e-30))
    energy_residual = float(np.sqrt(np.mean((2*work-estimate*q)**2))/max(abs(2*work[-1]), 1e-30))
    return dict(status="resolved_estimate" if len(a) >= min_samples and estimate > 0 else "low_resolution_estimate",
                estimate=estimate, end_index=end, samples=len(a), peak_index=peak,
                end_fraction=end_fraction, end_coordinate=float(a[-1]), initial_work=float(work[-1]),
                secant_energy_estimate=float(2*work[-1]/a[-1]**2),
                linearity_residual_over_stress_span=residual,
                energy_fit_residual=energy_residual, baseline_x=float(x[0]), baseline_y=float(baseline_response),
                observed_initial_response=float(y[0]),
                definition="E=2*integral(response dcoordinate)/coordinate_end^2 through declared filtered-peak fraction; physical unloaded origin")


def qualify_stiffness(time, x, y, natural_frequency_Hz):
    """Frozen40% interval plus30/50% and sampling-cutoff sensitivity screens."""
    (fx, fy), filter_info = archive_filter(time, x, y, natural_frequency_Hz)
    primary = initial_work_stiffness(fx, fy)
    windows = {str(f): initial_work_stiffness(fx, fy, f) for f in (.3, .4, .5)}
    cutoffs = {}
    for fraction in (.2, 1/3, .4, .6):
        (a, b), info = archive_filter(time, x, y, natural_frequency_Hz, fraction,
                                      respect_resolved_physical=False)
        cutoffs[str(fraction)] = dict(initial_work_stiffness(a, b), cutoff_Hz=info["cutoff_Hz"])
    value = primary.get("estimate")
    def spread(records):
        vals = [v.get("estimate") for v in records.values()]
        if value is None or value <= 0 or any(v is None or v <= 0 for v in vals):
            return None
        return float(max(abs(np.asarray(vals)/value-1)))
    window_spread, cutoff_spread = spread(windows), spread(cutoffs)
    flags = []
    if primary["status"] != "resolved_estimate": flags.append(primary["status"])
    if window_spread is None or window_spread > .05: flags.append("window_sensitive_over5percent")
    if cutoff_spread is None or cutoff_spread > .03: flags.append("cutoff_sensitive_over3percent")
    # Oscillatory response is the reason for integrating work. Pointwise force
    # residual is retained as a diagnostic, not used to reject an otherwise
    # stable work-equivalent slope. Systematic curvature survives integration.
    if primary.get("energy_fit_residual", float("inf")) > .03:
        flags.append("systematic_energy_curvature_over3percent")
    notes=[]
    if primary.get("linearity_residual_over_stress_span", float("inf")) > .05:
        notes.append("oscillatory_response_energy_stability_required")
    return dict(primary, quality="passes_frozen_numerical_screens" if not flags else "review_required",
                flags=flags, quality_notes=notes, window_relative_spread=window_spread, cutoff_relative_spread=cutoff_spread,
                windows=windows, cutoff_sensitivity=cutoffs, filter=filter_info,
                interpretation="Initial work-equivalent curve stiffness; no isotropy classification or material-tensor claim")


def records(root):
    folder = Path(root)/"audit/curve-processing/paired-examples"
    selected = json.loads((folder/"selection.json").read_text())
    native = json.loads((folder/"native-index.json").read_text())
    return list(selected["periodic"])+native


def numerical_metrics(x, y):
    peak = int(np.argmax(y))
    return dict(peak=float(np.max(y)), roughness=float(np.sqrt(np.mean(np.diff(y, 2)**2))/max(np.ptp(y), 1e-30)),
                full_work=float(np.trapz(y, x)), peak_index=peak)


def read_columns(path, names):
    with Path(path).open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    return [np.array([float(row[k]) for row in rows]) for k in names]


def ft_gold_controls(root):
    """Existing source-exact elastic controls; no solver submission."""
    audit=Path(root)/"audit"
    with(Path(root)/"case-index.csv").open(newline="") as stream:
        cases={r["case_id"]:r for r in csv.DictReader(stream)}
    modal=json.loads((audit/"modal/iso-periodic/source-frequency-map.json").read_text())
    output=[]
    for c in json.loads((audit/"controls/compliance-summary.json").read_text()):
        if c["source_sha256"] not in modal:continue
        case=cases[c["case_id"]];odb=Path(case["odb"])
        path=audit/"history"/odb.parent/odb.stem/"native_curve.csv"
        t,x,y=read_columns(path,("time_s","U2_mm","RF2_N"))
        fit=qualify_stiffness(t,x,y,modal[c["source_sha256"]]["natural_frequency_Hz"])
        gold=c["static_stiffness_N_mm"]
        output.append(dict(job=c["job"],native=str(path),native_sha256=sha(path),gold_k_N_mm=gold,
                           source_inp=c["source_inp"],source_inp_sha256=c["source_sha256"],
                           curve_estimate_N_mm=fit["estimate"],relative_error_percent=100*(fit["estimate"]/gold-1),qualification=fit))
    path=audit/"fcc-calibration/native/native-curve.csv"
    t,x,y=read_columns(path,("time_s","pin_U2_mm","pin_RF2_N"))
    receipt=audit/"fcc-calibration/fcc_el_a0/result.json"
    control=json.loads(receipt.read_text());gold=control["static_stiffness_N_mm"]
    fit=qualify_stiffness(t,x,y,694.95)
    output.append(dict(job="FCC_actualCT",native=str(path),native_sha256=sha(path),gold_k_N_mm=gold,
                       elastic_control_receipt=str(receipt),elastic_control_receipt_sha256=sha(receipt),
                       curve_estimate_N_mm=fit["estimate"],relative_error_percent=100*(fit["estimate"]/gold-1),qualification=fit))
    return output


def dense_reference_comparison(root=ROOT):
    """Compare archive adaptation with genuinely dense, physical2fn filtering.

    Sampling comparison uses the same dense source at both resolutions, so it
    does not attribute the tiny source-run difference to filtering. Material
    initiation time comes from an independent unfiltered field event.
    """
    root=Path(root);audit=root/"audit";out=audit/"curve-processing/accepted-filter-study"
    folder=audit/"fcc-calibration/initiation-resolution/fcc_init_dense"
    receipt=json.loads((folder/"resolved-event.json").read_text())
    dense=folder/"dense-native-curve.csv";original=audit/"fcc-calibration/native/native-curve.csv"
    t,x,y=read_columns(dense,("time_s","U2_mm","RF2_N"))
    ot,ox,oy=read_columns(original,("time_s","pin_U2_mm","pin_RF2_N"))
    endpoint=receipt["event"]["previous_field"]["time_s"]
    # Use common archived observation times, with the event bracket unchanged.
    ct=ot[ot<=endpoint+1e-9];cx=np.interp(ct,t,x);cy=np.interp(ct,t,y)
    (dx,dy),di=archive_filter(t,x,y,694.95)
    (archive_x,archive_y),ai=archive_filter(ct,cx,cy,694.95)
    (oldx,oldy),oi=archive_filter(ot,ox,oy,694.95)
    sampled_dense_x=np.interp(ct,t,dx);sampled_dense_y=np.interp(ct,t,dy)
    def measured(a,b):return dict(force_N=float(b[-1]),opening_mm=float(a[-1]),work_N_mm=float(np.trapz(b,a)))
    methods={"dense_physical2fn_then_sampled":measured(sampled_dense_x,sampled_dense_y),
             "same_source_coarse_archive_adaptation":measured(archive_x,archive_y),
             "same_source_coarse_raw":measured(cx,cy),
             "original_coarse_archive_adaptation":measured(oldx[:len(ct)],oldy[:len(ct)])}
    target=methods["dense_physical2fn_then_sampled"]
    for v in methods.values():
        v["force_difference_percent_vs_dense"]=100*(v["force_N"]/target["force_N"]-1)
        v["work_difference_percent_vs_dense"]=100*(v["work_N_mm"]/target["work_N_mm"]-1)
    report=dict(source=str(dense),source_sha256=sha(dense),event_receipt_sha256=sha(folder/"resolved-event.json"),
                material_event_time_s=endpoint,comparison_time_s=float(ct[-1]),dense_samples=len(t),coarse_samples=len(ct),
                physical_cutoff_Hz=di["cutoff_Hz"],archive_cutoff_Hz=ai["cutoff_Hz"],methods=methods,
                limitation="One FCC prefix verifies this adaptation on this case; it does not reconstruct unavailable histories in other cases")
    out.mkdir(parents=True,exist_ok=True)
    (out/"dense-FCC-reference-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.3))
    for panel in axes:
        mask=t<=endpoint
        panel.plot(x[mask],y[mask]/1000,color=".75",lw=.6,label="Dense raw response")
        panel.plot(dx[mask],dy[mask]/1000,color="#0000ff",lw=1.2,label=r"Dense causal Butterworth, $2f_n=1389.9$ Hz")
        panel.plot(archive_x,archive_y/1000,color="#ff7f0e",lw=1.1,label=r"Saved-rate adaptation, $f_s/6=333.3$ Hz")
        panel.set_xlabel("Load-line displacement (mm)");panel.set_ylabel("Force (kN)");panel.grid(alpha=.18)
    axes[1].set_xlim(5.8,6.5);axes[1].set_ylim(75,95)
    axes[0].legend(fontsize=8,frameon=False)
    fig.tight_layout()
    for ext in("png","svg"):fig.savefig(out/("dense-FCC-filter-comparison."+ext),dpi=210)
    plt.close(fig)
    return report


def population_hex_replay(root=ROOT):
    """Read-only qualification of retained raw transfer prefixes, not ML curves.

    Membership follows the existing saved property index; no saved property is
    replaced. Transfer timestamps are unavailable, so the source's declared
    0.0005s output schedule is recorded explicitly as an assumption.
    """
    from concurrent.futures import ThreadPoolExecutor
    from validation_curve_processing import transfer_prefix
    root=Path(root);out=root/"audit/curve-processing/accepted-filter-study"
    registry=json.loads((root/"audit/curve-processing/periodic-cutoffs.json").read_text())
    fn=next(r["natural_frequency_Hz"] for r in registry["families"] if r["topology"]=="hex" and r["task"]=="UT")
    work=[];membership=[]
    for design,base,suffix in(("full",Path("Z:/p1/data/Ti/disNodes/0.2/hex"),"all"),
                             ("mechanism",Path("Z:/p1/data/Ti/disNodes/Target-xs/0.2/hex"),"xs")):
        prop=base/"MLdata/UT-disNodes-allProps.csv"
        with prop.open(newline="") as stream:
            ids=[int(r[0]) for r in list(csv.reader(stream))[1:] if r and int(r[0])>0]
        membership.append(dict(design=design,path=str(prop),sha256=sha(prop),retained_ids=len(ids)))
        for identity in ids:
            path=base/"transfer"/("OUT-Ductile-hex-20-20disNodes-lhs-"+suffix+"-"+str(identity)+".csv")
            work.append((design,identity,path))
    def process(item):
        design,identity,path=item
        try:
            points,support=transfer_prefix(path)
            time=np.arange(len(points))*.0005
            fit=qualify_stiffness(time,points[:,0],points[:,1],fn)
            return dict(design=design,id=identity,path=str(path),source_sha256=sha(path),support=support,stiffness=fit)
        except (ValueError,OSError,IndexError) as error:
            return dict(design=design,id=identity,path=str(path),error=str(error))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(process,work))
    report=dict(source="Raw transfer responses; not previously smoothed MLdata outputs",membership=membership,
                time_basis="Declared common output schedule0.0005s; original positions retained, no stretching or tail imputation",
                native_timestamp_proof=False,no_source_or_saved_property_writes=True,records=rows,
                counts=dict(total=len(rows),passed=sum(r.get("stiffness",{}).get("quality")=="passes_frozen_numerical_screens" for r in rows),
                            flagged=sum(r.get("stiffness",{}).get("quality")=="review_required" for r in rows),errors=sum("error"in r for r in rows)))
    out.mkdir(parents=True,exist_ok=True)
    (out/"hex-population-readonly-qualification.json").write_text(json.dumps(report,indent=2)+"\n")
    columns=[dict(design=r["design"],id=r["id"],estimate=r.get("stiffness",{}).get("estimate"),
                  quality=r.get("stiffness",{}).get("quality","error"),flags=";".join(r.get("stiffness",{}).get("flags",[])),
                  window_spread=r.get("stiffness",{}).get("window_relative_spread"),
                  cutoff_spread=r.get("stiffness",{}).get("cutoff_relative_spread"),
                  energy_curvature=r.get("stiffness",{}).get("energy_fit_residual"))for r in rows]
    with(out/"hex-population-readonly-stiffness.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(columns[0]));writer.writeheader();writer.writerows(columns)
    flagged=[r for r in rows if r.get("stiffness",{}).get("quality")=="review_required"]
    if flagged:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncols=3;nrows=int(np.ceil(len(flagged)/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(12,3.3*nrows),squeeze=False)
        for ax,row in zip(axes.flat,flagged):
            points,_=transfer_prefix(row["path"]);time=np.arange(len(points))*.0005
            (x,y),_=archive_filter(time,points[:,0],points[:,1],fn)
            end=row["stiffness"]["windows"]["0.5"]["end_index"]+2
            ax.plot(points[:end,0],points[:end,1],color=".7",lw=.8,label="Raw")
            ax.plot(x[:end],y[:end],color="#008000",lw=1.1,label="Butterworth")
            for fraction,style in(("0.3",":"),("0.4","-"),("0.5","--")):
                fit=row["stiffness"]["windows"][fraction]
                xx=np.array([0.,fit["end_coordinate"]]);ax.plot(xx,fit["estimate"]*xx,ls=style,lw=.9,label=str(int(float(fraction)*100))+"% work slope")
            ax.set_title(row["design"]+" ID"+str(row["id"])+" | window spread"+format(100*row["stiffness"]["window_relative_spread"],".2f")+"%",fontsize=9)
            ax.set_xlabel(r"Macroscopic strain, $\varepsilon_L$");ax.set_ylabel(r"Macroscopic stress, $\Sigma$ (MPa)");ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(flagged):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=7,frameon=False)
        fig.tight_layout()
        for ext in("png","svg"):fig.savefig(out/("hex-flagged-elastic-windows."+ext),dpi=180)
        plt.close(fig)
    return report


def prepare_holdout_controls(root=ROOT):
    """Prepare holdouts transactionally; existing evidence is immutable.

    All inputs are first rendered in a private staging directory. Every final
    target is compared before any final target is written. Repeating an
    identical preparation performs no writes to existing files.
    """
    import tempfile
    from validation_controls import prepare_uat_linear
    root=Path(root);study=root/"audit/curve-processing/accepted-filter-study"
    out=study/"holdout-controls";out.mkdir(parents=True,exist_ok=True)
    replay=study/"hex-population-readonly-qualification.json"
    population=json.loads(replay.read_text())
    selected=[r for r in population["records"] if r.get("stiffness",{}).get("quality")=="review_required"]
    expected={("full",21),("mechanism",219),("mechanism",274),("mechanism",643),("mechanism",670),
              ("mechanism",698),("mechanism",699),("mechanism",782),("mechanism",901)}
    if {(r["design"],r["id"]) for r in selected}!=expected:
        raise ValueError("Holdout identity changed; do not silently alter the frozen validation set")
    jobs=[];predictions=[];targets={}
    with tempfile.TemporaryDirectory(prefix=".holdout-stage-",dir=study) as temporary:
        staging=Path(temporary).resolve()
        if not staging.is_relative_to(study.resolve()):
            raise ValueError("Staging cleanup must remain inside the audit study")
        for row in selected:
            sources=_holdout_sources(row)
            if not sources:raise FileNotFoundError("No exact archived source for "+str(row["id"]))
            if len({sha(p) for p in sources})!=1:
                raise ValueError("Different archived inputs share identity: "+str(row["id"]))
            source=sorted(sources,key=lambda p:str(p))[0]
            name="uat_hex_holdout_"+row["design"]+str(row["id"])
            control=prepare_uat_linear(source,staging,name,travel=1e-4)
            target_input=out/name/(name+".inp")
            targets[target_input]=Path(control["input"]).read_text(encoding="ascii")
            control["input"]=str(target_input)
            targets[out/name/"source.json"]=json.dumps(control,indent=2,allow_nan=False)
            jobs.append(dict(job=name,input=str(target_input),input_sha256=control["input_sha256"],cpus=8,
                             required_outputs=[".odb",".dat",".sta"]))
            predictions.append(dict(design=row["design"],id=row["id"],job=name,source_inp=str(source),
                                    source_inp_sha256=control["source_sha256"],identical_source_paths=[str(p) for p in sources],
                                    frozen_curve_estimate_MPa=row["stiffness"]["estimate"],quality_flags=row["stiffness"]["flags"]))
    manifest=dict(abaqus="C:/SIMULIA/Commands/abaqus.bat",queue_root=str(out/"queue"),jobs=jobs)
    targets[out/"manifest.json"]=json.dumps(manifest,indent=2)+"\n"
    freeze=dict(purpose="Independent validation of a frozen curve-derived estimator; not replacement of curve values by FE values",
                estimator_file=str(Path(__file__).resolve()),estimator_sha256=sha(__file__),
                source_replay=str(replay),source_replay_sha256=sha(replay),
                rule="Physical2fn if resolved, otherwise fs/6; paired causal order2; initial40%work slope;30/50%window check",
                holdout_acceptance="Report every holdout error against5%target; no parameter tuning from these results",
                no_solver_launched=True,predictions=predictions)
    frozen_path=out/"frozen-predictions.json"
    if frozen_path.exists():
        previous=json.loads(frozen_path.read_text())
        if previous["estimator_sha256"]!=freeze["estimator_sha256"]:
            # The original file is evidence. A reviewed maintenance-only code
            # change needs its own receipt; it never rewrites the predictions.
            expected_receipt=dict(frozen_predictions_sha256=sha(frozen_path),
                original_estimator_sha256=previous["estimator_sha256"],
                reviewed_estimator_sha256=freeze["estimator_sha256"],
                numerical_core_sha256=numerical_core_sha256())
            paths=[out/"reviewed-code-update.json"]+sorted(out.glob("reviewed-code-update-*.json"))
            receipts=[json.loads(p.read_text()) for p in paths if p.is_file()]
            if not any(all(r.get(k)==v for k,v in expected_receipt.items()) for r in receipts):
                raise ValueError("Frozen estimator differs; reviewed code-update receipt required")
            freeze["estimator_sha256"]=previous["estimator_sha256"]
    targets[frozen_path]=json.dumps(freeze,indent=2)+"\n"
    _write_immutable_targets(targets)
    return manifest


def numerical_core_sha256():
    """Fingerprint the frozen filter/estimator mathematics separately from CLI."""
    import inspect
    source="\n".join(inspect.getsource(f) for f in (archive_filter,initial_work_stiffness,qualify_stiffness))
    return hashlib.sha256(source.encode()).hexdigest()


def _holdout_sources(row):
    if row["design"]=="full":
        base=Path("Z:/p1/HPCarchive/Ti/data/disNodes/0.2/hex");suffix="all"
    else:
        base=Path("Z:/p1/HPCarchive/Ti/data/disNodes/Target-xs/0.2/hex");suffix="xs"
    name="Ductile-hex-20-20disNodes-lhs-"+suffix+"-"+str(row["id"])+".inp"
    return [job/"zip"/name for job in base.iterdir() if job.is_dir() and (job/"zip"/name).is_file()]


def _write_immutable_targets(targets):
    """Preflight the complete target set before creating any missing file."""
    for path,text in targets.items():
        if path.exists() and path.read_text(encoding="utf-8")!=text:
            raise ValueError("Refuse different existing holdout evidence: "+str(path))
    for path,text in targets.items():
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",encoding="utf-8") as stream:stream.write(text)


def evaluate_holdouts(predictions,controls):
    """Evaluate every frozen prediction; never replace or tune a prediction."""
    import math
    tolerance=.05
    if not predictions:raise ValueError("Nonempty complete frozen holdout set required")
    if len({r["job"] for r in predictions})!=len(predictions) or len({r["job"] for r in controls})!=len(controls):
        raise ValueError("Duplicate holdout identity")
    by_job={r["job"]:r for r in controls}
    if set(by_job)!={r["job"] for r in predictions}:
        raise ValueError("Complete frozen holdout set required; no missing or additional controls")
    rows=[]
    for prediction in predictions:
        control=by_job[prediction["job"]]
        if control["source_sha256"]!=prediction["source_inp_sha256"]:
            raise ValueError("Holdout source differs for "+prediction["job"])
        if control.get("state")!="completed_and_extracted" or control.get("read_only") is not True:
            raise ValueError("Receipt-verified extraction required for "+prediction["job"])
        estimate=prediction["frozen_curve_estimate_MPa"];reference=control["E_apparent_MPa"]
        if not all(math.isfinite(v) and v>0 for v in(estimate,reference)):
            raise ValueError("Positive finite apparent moduli required for "+prediction["job"])
        error=estimate/reference-1
        numerical={k:control[k] for k in("force_balance_relative","work_energy_relative","clamp_work_energy_relative")}
        if not all(math.isfinite(v) and 0<=v<=1e-5 for v in numerical.values()):
            raise ValueError("Independent control balance/energy gate failed for "+prediction["job"])
        rows.append(dict(design=prediction["design"],id=prediction["id"],job=prediction["job"],
                         source_inp_sha256=prediction["source_inp_sha256"],
                         frozen_curve_estimate_MPa=estimate,elastic_control_E_apparent_MPa=reference,
                         relative_error_percent=100*error,within_five_percent=abs(estimate-reference)<=tolerance*reference,
                         original_curve_quality_flags=prediction["quality_flags"],
                         control_numerical_checks=numerical))
    accepted=all(r["within_five_percent"] for r in rows)
    return dict(holdouts=len(rows),acceptance_tolerance_percent=100*tolerance,
                all_holdouts_within_target=accepted,max_absolute_error_percent=max(abs(r["relative_error_percent"]) for r in rows),
                verdict="accept_frozen_curve_estimates_with_independent_holdout_qualification" if accepted else
                        "retain_sensitivity_flags_independent_holdout_target_not_met",
                no_estimates_replaced_or_retuned=True,records=rows)


def compare_holdout_controls(root=ROOT):
    """Read completed receipt-verified extraction and persist immutable verdict."""
    root=Path(root);out=root/"audit/curve-processing/accepted-filter-study/holdout-controls"
    frozen_path=out/"frozen-predictions.json";summary_path=out/"uat-linear-summary.json"
    frozen=json.loads(frozen_path.read_text());controls=json.loads(summary_path.read_text())
    predictions=frozen["predictions"]
    manifest=json.loads((out/"manifest.json").read_text())
    if len(predictions)!=9 or {r["job"] for r in predictions}!={r["job"] for r in manifest["jobs"]}:
        raise ValueError("Original nine-job holdout manifest required")
    report=evaluate_holdouts(predictions,controls)
    report.update(frozen_predictions_sha256=sha(frozen_path),extracted_controls_sha256=sha(summary_path),
                  frozen_original_estimator_sha256=frozen["estimator_sha256"],numerical_core_sha256=numerical_core_sha256(),
                  source_rule=frozen["rule"],reference_gauge="Apparent modulus: identical original clamp/body-height/gross-area convention",
                  scope="Independent source-exact elastic UAT controls qualify curve-derived stiffness only; not nonlinear fracture validation")
    _write_immutable_targets({out/"holdout-comparison.json":json.dumps(report,indent=2)+"\n"})
    path=out/"holdout-comparison.png"
    if not path.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4.4));rows=report["records"]
        ax.bar(np.arange(len(rows)),[r["relative_error_percent"] for r in rows],
               color=["#ff7f0e" if r["design"]=="full" else "#1f77b4" for r in rows])
        ax.axhline(0,color=".3",lw=.8)
        for value in(-5,5):ax.axhline(value,color=".5",ls="--",lw=.8)
        ax.set_xticks(np.arange(len(rows)),[r["design"]+"\nID"+str(r["id"]) for r in rows],fontsize=8)
        ax.set_ylabel("Frozen curve estimate minus elastic-control modulus (%)")
        ax.grid(axis="y",alpha=.15);fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)
    return report


def select_accepted_hex_stiffness(population,frozen,controls,verified_jobs):
    """Apply the preselected quality flag, never the subsequent holdout error.

    Every preflagged specimen uses its own elastic control, including controls
    whose curve estimates happened to pass the holdout error target.
    """
    import math
    predictions={(r["design"],r["id"]):r for r in frozen["predictions"]}
    by_job={r["job"]:r for r in controls}
    flagged={(r["design"],r["id"]) for r in population["records"] if r["stiffness"]["quality"]=="review_required"}
    if flagged!=set(predictions):raise ValueError("Frozen preflagged selection differs from source replay")
    if len(predictions)!=len(frozen["predictions"]) or len(by_job)!=len(controls):
        raise ValueError("Duplicate accepted stiffness identity")
    needed={r["job"] for r in predictions.values()}
    if not needed.issubset(verified_jobs) or set(by_job)!=needed:
        raise ValueError("Missing or unverified preselected elastic fallback")
    rows=[]
    for record in population["records"]:
        key=record["design"],record["id"];fit=record["stiffness"]
        if key in predictions:
            prediction=predictions[key];control=by_job[prediction["job"]]
            if (control["source_sha256"]!=prediction["source_inp_sha256"] or
                    fit["estimate"]!=prediction["frozen_curve_estimate_MPa"]):
                raise ValueError("Frozen fallback source/prediction differs")
            value=control["E_apparent_MPa"];method="source_exact_linear_UAT_preflagged_fallback"
        else:
            if fit["quality"]!="passes_frozen_numerical_screens":raise ValueError("Unqualified curve has no preselected fallback")
            value=fit["estimate"];control=None;method="filtered_initial_work_curve"
        if not math.isfinite(value) or value<=0:raise ValueError("Positive finite accepted modulus required")
        rows.append(dict(design=record["design"],id=record["id"],E_apparent_MPa=value,estimator=method,
                         raw_transfer=record["path"],raw_transfer_sha256=record["source_sha256"],
                         original_curve_estimate_MPa=fit["estimate"],original_curve_quality=fit["quality"],
                         original_curve_flags=fit["flags"],window_relative_spread=fit["window_relative_spread"],
                         cutoff_relative_spread=fit["cutoff_relative_spread"],
                         fallback_job=control["job"] if control else None,
                         fallback_source_inp=control["source_inp"] if control else None,
                         fallback_source_inp_sha256=control["source_sha256"] if control else None,
                         fallback_input_sha256=control["input_sha256"] if control else None,
                         fallback_odb_sha256=control["odb_sha256"] if control else None,
                         fallback_delivery_sha256=control["delivery_sha256"] if control else None))
    if len({(r["design"],r["id"]) for r in rows})!=len(rows):raise ValueError("Duplicate population identity")
    return rows


def accepted_hex_stiffness(root=ROOT):
    """Write an external accepted catalogue; never update Props or MLdata.

    Membership comes from quality flags frozen before FE results. After the
    holdout review, the final policy uses1964 curve estimates and all nine
    elastic controls uniformly, rather than choosing only the three errors
    exceeding the holdout target.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor
    from validation_controls import verified_control_delivery
    root=Path(root);study=root/"audit/curve-processing/accepted-filter-study";out=study/"holdout-controls"
    frozen_path=out/"frozen-predictions.json";freeze=json.loads(frozen_path.read_text())
    replay_path=study/"hex-population-readonly-qualification.json"
    if sha(replay_path)!=freeze["source_replay_sha256"]:raise ValueError("Frozen source replay changed")
    population=json.loads(replay_path.read_text())
    if len(population["records"])!=1973 or len(freeze["predictions"])!=9:
        raise ValueError("Original1973-specimen population and nine preselected fallbacks required")
    # The original freeze is immutable; a reviewed maintenance receipt binds
    # its original module hash to the unchanged numerical function sources.
    receipts=[out/"reviewed-code-update.json"]+sorted(out.glob("reviewed-code-update-*.json"))
    if not any(json.loads(p.read_text()).get("frozen_predictions_sha256")==sha(frozen_path) and
               json.loads(p.read_text()).get("numerical_core_sha256")==numerical_core_sha256()
               for p in receipts if p.is_file()):
        raise ValueError("Original freeze/numerical-core verification required")
    summary_path=out/"uat-linear-summary.json";controls=json.loads(summary_path.read_text())
    queue=out/"queue";state=json.loads((queue/"state.json").read_text())
    completed={r["job"] for r in state["jobs"] if r["status"]=="completed"}
    verified=set();delivered_files=0
    for control in controls:
        odb,receipt_path=verified_control_delivery(control,queue,completed)
        if (sha(odb)!=control["odb_sha256"] or sha(receipt_path)!=control["delivery_sha256"] or
                control.get("state")!="completed_and_extracted" or control.get("read_only") is not True):
            raise ValueError("Elastic fallback extraction/delivery identity differs")
        delivery=json.loads(receipt_path.read_text())
        for item in delivery["files"]:
            path=receipt_path.parent/item["file"]
            if path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]:
                raise ValueError("Elastic fallback delivery changed: "+str(path))
            delivered_files+=1
        checks=[control[k] for k in("force_balance_relative","work_energy_relative","clamp_work_energy_relative")]
        if not all(np.isfinite(v) and 0<=v<=1e-5 for v in checks):raise ValueError("Elastic fallback numerical gate failed")
        verified.add(control["job"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw_matches=list(pool.map(lambda r:sha(r["path"])==r["source_sha256"],population["records"]))
    if not all(raw_matches):raise ValueError("Source raw transfer changed since frozen qualification")
    rows=select_accepted_hex_stiffness(population,freeze,controls,verified)
    curve_count=sum(r["estimator"]=="filtered_initial_work_curve" for r in rows)
    if curve_count!=1964:raise ValueError("Frozen1964curve/ninefallback selection changed")
    report=dict(policy="One quality-controlled workflow: stable curve estimates; all preflagged cases use source-exact elastic controls",
                counts=dict(total=len(rows),curve=curve_count,elastic_fallback=len(rows)-curve_count),
                no_Props_or_MLdata_modified=True,no_original_curve_estimates_modified=True,
                no_selection_based_on_holdout_error=True,reference_gauge="Apparent modulus from original clamp travel/body height/gross area",
                frozen_predictions_sha256=sha(frozen_path),source_replay_sha256=sha(replay_path),
                extracted_controls_sha256=sha(summary_path),numerical_core_sha256=numerical_core_sha256(),
                verified_raw_transfers=len(raw_matches),verified_fallback_controls=len(verified),verified_delivery_files=delivered_files,
                records=rows)
    csv_buffer=io.StringIO(newline="");writer=csv.DictWriter(csv_buffer,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader()
    for row in rows:writer.writerow(dict(row,original_curve_flags=";".join(row["original_curve_flags"])))
    _write_immutable_targets({study/"accepted-hex-stiffness.json":json.dumps(report,indent=2)+"\n",
                              study/"accepted-hex-stiffness.csv":csv_buffer.getvalue()})
    return report


def uniform_fs6_curve(time,x,y):
    """Sampling-rule sensitivity only; frozen physical/archive default stays unchanged."""
    t=np.asarray(time,float)
    if t.ndim!=1 or len(t)<6 or not np.isfinite(t).all() or t[-1]<=t[0]:
        raise ValueError("At least six finite increasing native observation times required")
    cutoff=(len(t)-1)/(6*(t[-1]-t[0]))
    fx,info=filter_force(t,np.asarray(x,float),cutoff,order=2,zero_phase=False)
    fy,_=filter_force(t,np.asarray(y,float),cutoff,order=2,zero_phase=False)
    info.update(policy="uniform_fs6_comparison_not_frozen_default",causal=True,paired=True)
    return (fx,fy),info


def response_endpoint_metrics(time,x,y):
    """Recorded post-peak crossings; absent thresholds remain explicitly censored."""
    from validation_curve_processing import first_crossing
    t,x,y=[np.asarray(a,float) for a in(time,x,y)]
    result=numerical_metrics(x,y);result["peak_time_s"]=float(t[result["peak_index"]])
    for fraction,name in((.25,"ductility25"),(.01,"work1")):
        end=first_crossing(y,fraction)
        result[name+"_observed"]=end is not None
        result[name+"_index"]=end
        result[name+"_time_s"]=float(t[end]) if end is not None else None
        result[name+"_coordinate"]=float(x[end]) if end is not None else None
        result[name+"_work"]=float(np.trapz(y[:end+1],x[:end+1])) if end is not None else None
    return result


def _relative_percent(after,before):
    return 100*(after/before-1) if before is not None and after is not None and before!=0 else None


def _fixed_curve_metrics(x,y,end):
    return dict(index=int(end),load=float(y[end]),work=float(np.trapz(y[:end+1],x[:end+1])))


def compare_uniform_fs6(root=ROOT):
    """Compare one sampling cutoff with the frozen conditional rule, without promotion."""
    root=Path(root);study=root/"audit/curve-processing/accepted-filter-study"
    out=study/"uniform-fs6-comparison";out.mkdir(parents=True,exist_ok=True)
    registry_path=root/"audit/curve-processing/periodic-cutoffs.json"
    registry={(r["topology"],r["task"]):r for r in json.loads(registry_path.read_text())["families"]}
    event_path=root/"audit/fcc-calibration/initiation-resolution/fcc_init_dense/resolved-event.json"
    fcc_event=json.loads(event_path.read_text())
    rows=[];arrays={}
    for record in records(root):
        path=Path(record["native"])
        if sha(path)!=record["native_sha256"]:raise ValueError("Changed native curve")
        with np.load(path,allow_pickle=False) as data:t,x,y=[data[k] for k in("time","x","y")]
        fn=registry[record["topology"],record["task"]]["natural_frequency_Hz"]
        (cx,cy),current_info=archive_filter(t,x,y,fn)
        (ux,uy),uniform_info=uniform_fs6_curve(t,x,y)
        legacy=np.asarray(smooth(smooth(y)))
        methods={name:response_endpoint_metrics(t,a,b) for name,a,b in
                 (("raw",x,y),("current_rule",cx,cy),("uniform_fs6",ux,uy),("legacy_double_MA",x,legacy))}
        current=methods["current_rule"];uniform=methods["uniform_fs6"]
        delta={key+"_percent":_relative_percent(uniform[key],current[key]) for key in
               ("peak","full_work","ductility25_coordinate","work1_work")}
        for name in("ductility25","work1"):
            delta[name+"_index_shift"]=(uniform[name+"_index"]-current[name+"_index"]
                                            if uniform[name+"_observed"] and current[name+"_observed"] else None)
            delta[name+"_crossing_observation_changed"]=uniform[name+"_observed"]!=current[name+"_observed"]
        material=None
        if record["task"]=="FT" and record["topology"]=="FCC" and record["design"]=="periodic":
            time=fcc_event["event"]["previous_field"]["time_s"]
            end=int(np.searchsorted(t,time+1e-9,side="right")-1)
            a,b=_fixed_curve_metrics(cx,cy,end),_fixed_curve_metrics(ux,uy,end)
            material=dict(criterion=fcc_event["criterion"],event_receipt=str(event_path),event_sha256=sha(event_path),
                          endpoint_time_s=float(t[end]),current=a,uniform=b,
                          load_change_percent=_relative_percent(b["load"],a["load"]),
                          work_change_percent=_relative_percent(b["work"],a["work"]))
        old_fit=initial_work_stiffness(cx,cy);new_fit=initial_work_stiffness(ux,uy)
        item={k:record[k] for k in("topology","task","design","id")}
        item.update(native=str(path),native_sha256=sha(path),samples=len(t),current_cutoff_Hz=current_info["cutoff_Hz"],
                    uniform_cutoff_Hz=uniform_info["cutoff_Hz"],methods=methods,uniform_minus_current=delta,
                    fixed_material_endpoint=material,current_initial_work_slope=old_fit["estimate"],
                    uniform_initial_work_slope=new_fit["estimate"],
                    initial_work_slope_change_percent=_relative_percent(new_fit["estimate"],old_fit["estimate"]))
        rows.append(item);arrays[record["topology"],record["task"],record["design"],record["id"]]=(x,y,cx,cy,ux,uy,legacy)
    gold,fixed_material=_uniform_gold_controls(root)
    changed=[r for r in rows if abs(r["uniform_cutoff_Hz"]-r["current_cutoff_Hz"])>1e-6]
    report=dict(policy="Uniformfs/6 comparison; frozen filter/estimator defaults, accepted hex catalogue and final figures unchanged",
                order=2,phase="causal, paired coordinate/response",source_basis=SOURCE,
                explanation="fs/6 is a sampling rule, not twice specimen natural frequency. Postprocessing cannot recover pre-output filtering or undo existing aliasing.",
                periodic_registry_sha256=sha(registry_path),native_histories=len(rows),changed_histories=len(changed),
                existing_hex_catalogue_unchanged="All hexUT cutoffs already equalfs/6; no accepted catalogue rewritten",
                records=rows,elastic_controls=gold,fixed_material_controls=fixed_material)
    _write_immutable_targets({out/"comparison.json":json.dumps(report,indent=2)+"\n"})
    flat=[{k:r[k] for k in("topology","task","design","id","current_cutoff_Hz","uniform_cutoff_Hz","initial_work_slope_change_percent")}
          |r["uniform_minus_current"] for r in rows]
    import io
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=list(flat[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(flat)
    _write_immutable_targets({out/"metrics.csv":stream.getvalue()})
    _plot_uniform_comparison(out,rows,arrays,gold)
    return report


def _uniform_gold_controls(root,selected_filter=None):
    audit=Path(root)/"audit";gold=[];fixed=[]
    native_records=json.loads((audit/"curve-processing/periodic-cutoff-native/index.json").read_text())
    registry={(r["topology"],r["task"]):r for r in json.loads((audit/"curve-processing/periodic-cutoffs.json").read_text())["families"]}
    for control in json.loads((audit/"controls/uat-initial-response/uat-linear-summary.json").read_text()):
        native=next(r for r in native_records if r["task"]=="UT" and r["source_inp"].lower()==control["source_inp"].lower())
        if sha(native["native"])!=native["native_sha256"]:raise ValueError("Changed native elastic-control comparison source")
        with np.load(native["native"],allow_pickle=False) as data:t,x,y=[data[k] for k in("time","x","y")]
        fn=registry[native["topology"],"UT"]["natural_frequency_Hz"]
        (cx,cy),_=archive_filter(t,x,y,fn);old=initial_work_stiffness(cx,cy)
        (ux,uy),_=(uniform_fs6_curve(t,x,y) if selected_filter is None else selected_filter(t,x,y,fn))
        fit=initial_work_stiffness(ux,uy);reference=control["E_apparent_MPa"]
        gold.append(dict(job=control["job"],task="UT",reference=reference,
                         current_error_percent=_relative_percent(old["estimate"],reference),uniform_error_percent=_relative_percent(fit["estimate"],reference),
                         basis="AllUT current and uniform cutoffs are identicalfs/6"))
    with(Path(root)/"case-index.csv").open(newline="") as stream:cases={r["case_id"]:r for r in csv.DictReader(stream)}
    controls={r["job"]:r for r in json.loads((audit/"controls/compliance-summary.json").read_text())}
    modes=json.loads((audit/"modal/iso-periodic/source-frequency-map.json").read_text())
    for record in ft_gold_controls(root):
        if record["job"]=="FCC_actualCT":
            columns=("time_s","pin_U2_mm","pin_RF2_N");fn=694.95;event_path=None
        else:
            columns=("time_s","U2_mm","RF2_N");control=controls[record["job"]]
            fn=modes[control["source_sha256"]]["natural_frequency_Hz"]
            odb=Path(cases[control["case_id"]]["odb"]);event_path=audit/"failure"/odb.parent/odb.stem/"initiation.json"
        t,x,y=read_columns(record["native"],columns)
        (ux,uy),info=(uniform_fs6_curve(t,x,y) if selected_filter is None else selected_filter(t,x,y,fn))
        fit=initial_work_stiffness(ux,uy)
        gold.append(dict(job=record["job"],task="FT",reference=record["gold_k_N_mm"],
                         current_error_percent=record["relative_error_percent"],
                         uniform_error_percent=_relative_percent(fit["estimate"],record["gold_k_N_mm"]),
                         uniform_cutoff_Hz=info["cutoff_Hz"]))
        if event_path is not None and event_path.is_file():
            event=json.loads(event_path.read_text());point=event.get("prev_native_history")
            if not point:continue
            end=point["index"]
            (cx,cy),ci=archive_filter(t,x,y,fn)
            a,b=_fixed_curve_metrics(cx,cy,end),_fixed_curve_metrics(ux,uy,end)
            fixed.append(dict(job=record["job"],event_receipt=str(event_path),event_sha256=sha(event_path),
                              criterion=event["criterion"],endpoint_time_s=float(t[end]),
                              current_cutoff_Hz=ci["cutoff_Hz"],uniform_cutoff_Hz=info["cutoff_Hz"],current=a,uniform=b,
                              load_change_percent=_relative_percent(b["load"],a["load"]),
                              work_change_percent=_relative_percent(b["work"],a["work"])))
    return gold,fixed


def _plot_uniform_comparison(out,rows,arrays,gold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    for topology in COLORS:
        for task in("UT","FT"):
            fig,axes=plt.subplots(4,2,figsize=(13,13.7),squeeze=False)
            for ax,row in zip(axes.flat,[r for r in rows if r["topology"]==topology and r["task"]==task]):
                x,y,cx,cy,ux,uy,legacy=arrays[topology,task,row["design"],row["id"]]
                scale=1. if task=="UT" else .001
                ax.plot(x,y*scale,color=".72",lw=.75)
                ax.plot(x,legacy*scale,color="#775599",lw=1.,ls="--")
                ax.plot(cx,cy*scale,color=COLORS[topology],lw=1.5)
                ax.plot(ux,uy*scale,color="#ff7f0e",lw=1.05)
                name="Periodic" if row["design"]=="periodic" else("Full disorder" if row["design"]=="full" else"Mechanism-based")+", ID"+str(row["id"])
                ax.set_title(name,color={"periodic":".1","full":"#ff7f0e","mechanism":"#1f77b4"}[row["design"]],fontsize=10)
                ax.text(.98,.04,"Current: {:.1f} Hz; uniform: {:.1f} Hz".format(row["current_cutoff_Hz"],row["uniform_cutoff_Hz"]),
                        transform=ax.transAxes,ha="right",fontsize=8,bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
                ax.set_xlabel(r"Macroscopic strain, $\varepsilon_L$" if task=="UT" else"Load-line displacement (mm)")
                ax.set_ylabel(r"Macroscopic stress, $\Sigma$ (MPa)" if task=="UT" else"Force (kN)");ax.grid(alpha=.18)
            fig.suptitle(topology+" | "+task+" | Filter comparison",y=.995)
            fig.legend(handles=[Line2D([],[],color=".72",label="Raw"),Line2D([],[],color=COLORS[topology],label="Current cutoff rule"),
                                Line2D([],[],color="#ff7f0e",label=r"Uniform $f_s/6$"),Line2D([],[],color="#775599",ls="--",label="Legacy double moving average")],
                       loc="upper center",bbox_to_anchor=(.5,.977),ncol=4,frameon=False)
            fig.tight_layout(rect=(0,0,1,.944),h_pad=1.6)
            for ext in("png","svg"):fig.savefig(out/(topology+"-"+task.lower()+"-comparison."+ext),dpi=180)
            plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.8));positions=np.arange(len(gold))
    ax.bar(positions-.18,[r["current_error_percent"] for r in gold],width=.36,color="#1f77b4",label="Current rule")
    ax.bar(positions+.18,[r["uniform_error_percent"] for r in gold],width=.36,color="#ff7f0e",label=r"Uniform $f_s/6$")
    for value in(-5,0,5):ax.axhline(value,color=".5",lw=.7,ls="--" if value else"-")
    labels=[r["job"].removeprefix("stiffness_").removeprefix("uat_").removesuffix("_unit_load").replace("_","\n") for r in gold]
    ax.set_xticks(positions,labels,fontsize=7);ax.set_ylabel("Curve stiffness minus source-exact elastic control (%)")
    ax.legend(frameon=False);fig.tight_layout()
    for ext in("png","svg"):fig.savefig(out/("elastic-controls-comparison."+ext),dpi=200)
    plt.close(fig)


def final_archive_filter(time,x,y,natural_frequency_Hz):
    """Selected single archive rule; production migration remains separate.

    Retain the physical2fn cutoff when slower than the sampling rule, and cap
    faster cutoffs atfs/6. This never raises a slow physical cutoff. The frozen
    earlier archive_filter remains unchanged for historical qualification.
    """
    paired,info=archive_filter(time,x,y,natural_frequency_Hz,nyquist_fraction=1/3,
                               respect_resolved_physical=False)
    info.update(policy="selected_minimum_of_physical2fn_and_fs6",default_cap="fc=min(2fn,fs/6)",
                status="Selected replacement; controlled production migration pending")
    return paired,info


def compare_capped_policy(root=ROOT):
    """Assess the selected capped physical rule against the frozen old rule."""
    root=Path(root);study=root/"audit/curve-processing/accepted-filter-study"
    out=study/"capped-physical-fs6";out.mkdir(parents=True,exist_ok=True)
    registry={(r["topology"],r["task"]):r for r in json.loads((root/"audit/curve-processing/periodic-cutoffs.json").read_text())["families"]}
    event_path=root/"audit/fcc-calibration/initiation-resolution/fcc_init_dense/resolved-event.json"
    fcc_event=json.loads(event_path.read_text())
    rows=[];arrays={}
    for source in records(root):
        path=Path(source["native"])
        if sha(path)!=source["native_sha256"]:raise ValueError("Changed native comparison source")
        with np.load(path,allow_pickle=False) as data:t,x,y=[data[k] for k in("time","x","y")]
        fn=registry[source["topology"],source["task"]]["natural_frequency_Hz"]
        (cx,cy),ci=archive_filter(t,x,y,fn);(sx,sy),si=final_archive_filter(t,x,y,fn)
        metrics=response_endpoint_metrics(t,sx,sy);old=response_endpoint_metrics(t,cx,cy)
        delta={key+"_percent":_relative_percent(metrics[key],old[key]) for key in
               ("peak","full_work","ductility25_coordinate","work1_work")}
        for name in("ductility25","work1"):
            delta[name+"_index_shift"]=(metrics[name+"_index"]-old[name+"_index"]
                if metrics[name+"_observed"] and old[name+"_observed"] else None)
            delta[name+"_crossing_observation_changed"]=metrics[name+"_observed"]!=old[name+"_observed"]
        fit=initial_work_stiffness(sx,sy);windows=[initial_work_stiffness(sx,sy,f) for f in(.3,.4,.5)]
        spread=max(abs(w["estimate"]/fit["estimate"]-1) for w in windows)
        material=None
        if source["task"]=="FT" and source["topology"]=="FCC" and source["design"]=="periodic":
            time=fcc_event["event"]["previous_field"]["time_s"]
            end=int(np.searchsorted(t,time+1e-9,side="right")-1)
            current=_fixed_curve_metrics(cx,cy,end);selected=_fixed_curve_metrics(sx,sy,end)
            material=dict(criterion=fcc_event["criterion"],event_receipt=str(event_path),event_sha256=sha(event_path),
                          endpoint_time_s=float(t[end]),current=current)
            material.update(selected=selected,load_change_percent=_relative_percent(selected["load"],current["load"]),
                            work_change_percent=_relative_percent(selected["work"],current["work"]))
        row={k:source[k] for k in("topology","task","design","id","native","native_sha256","samples")}
        row.update(current_cutoff_Hz=ci["cutoff_Hz"],selected_cutoff_Hz=si["cutoff_Hz"],
                   methods={"raw":response_endpoint_metrics(t,x,y),"old_conditional_rule":old,"selected_capped":metrics,
                            "legacy_double_MA":response_endpoint_metrics(t,x,np.asarray(smooth(smooth(y))))},selected_minus_old=delta,
                   initial_work_slope=fit["estimate"],initial_work_slope_change_percent=_relative_percent(fit["estimate"],initial_work_stiffness(cx,cy)["estimate"]),
                   window30_40_50_relative_spread=spread,energy_curvature=fit["energy_fit_residual"],
                   passes_stiffness_window=fit["status"]=="resolved_estimate" and spread<=.05 and fit["energy_fit_residual"]<=.03,
                   fixed_material_endpoint=material)
        rows.append(row);arrays[source["topology"],source["task"],source["design"],source["id"]]=(x,y,cx,cy,sx,sy,np.asarray(smooth(smooth(y))))
    gold,fixed=_uniform_gold_controls(root,selected_filter=final_archive_filter)
    def relabel(record):return {k.replace("uniform","selected",1) if k.startswith("uniform") else k:
                                  (v.replace("uniform","selected") if isinstance(v,str) else v) for k,v in record.items()}
    gold=[relabel(r) for r in gold];fixed=[relabel(r) for r in fixed]
    for point in fixed:
        point["elastic_K_change_percent"]=point["load_change_percent"]
        point["elastic_J_change_percent"]=100*((1+point["load_change_percent"]/100)**2-1)
    changed=[r for r in rows if abs(r["selected_cutoff_Hz"]-r["current_cutoff_Hz"])>1e-6]
    report=dict(selected_policy="fc=min(2fn,fs/6), causal order2 paired filtering",
                status="Selected single replacement; no production values or final paper figures changed",
                old_rule_role="Historical comparison only; original numerical core remains frozen",
                uniform_fs6_role="Separate sensitivity study; raises slow physical cutoffs and is not the selected rule",
                native_histories=len(rows),changed_histories=len(changed),
                stiffness_window="One initial40%filtered-peak work interval forUTandFT, checked at30/50%",
                stiffness_windows_passed=sum(r["passes_stiffness_window"] for r in rows),
                allUT_and_accepted1973hex_catalogue_unchanged=True,records=rows,
                elastic_controls=gold,fixed_material_controls=fixed,
                physical_interpretation="Physical2fn is retained when resolved belowfs/6. fs/6 is an archive sampling cap, not a specimen mode or reconstruction of unavailable solver-rate data.")
    figure_manifest=_plot_capped_policy(out,study,rows,arrays)
    report["figures"]=figure_manifest
    _write_immutable_targets({out/"comparison.json":json.dumps(report,indent=2)+"\n"})
    return report


def _plot_capped_policy(out,study,rows,arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figures=[]
    def panel(ax,row):
        x,y,cx,cy,sx,sy,legacy=arrays[row["topology"],row["task"],row["design"],row["id"]]
        scale=1. if row["task"]=="UT" else .001
        ax.plot(x,y*scale,color=".72",lw=.75)
        ax.plot(x,legacy*scale,color="#775599",ls="--",lw=1.)
        ax.plot(cx,cy*scale,color=COLORS[row["topology"]],lw=1.5)
        ax.plot(sx,sy*scale,color="#ff7f0e",lw=1.05)
        ax.set_xlabel(r"Macroscopic strain, $\varepsilon_L$" if row["task"]=="UT" else"Load-line displacement (mm)")
        ax.set_ylabel(r"Macroscopic stress, $\Sigma$ (MPa)" if row["task"]=="UT" else"Force (kN)");ax.grid(alpha=.18)
        label="Periodic" if row["design"]=="periodic" else("Full disorder" if row["design"]=="full" else"Mechanism-based")+", ID"+str(row["id"])
        ax.set_title(row["topology"]+" | "+label,fontsize=10)
        ax.text(.98,.04,"Old: {:.1f} Hz; selected: {:.1f} Hz".format(row["current_cutoff_Hz"],row["selected_cutoff_Hz"]),
                transform=ax.transAxes,ha="right",fontsize=8,bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
    handles=[Line2D([],[],color=".72",label="Raw"),Line2D([],[],color="black",label="Old conditional rule"),
             Line2D([],[],color="#ff7f0e",label=r"Selected: $\min(2f_n,f_s/6)$"),Line2D([],[],color="#775599",ls="--",label="Legacy double moving average")]
    for topology in COLORS:
        for task in("UT","FT"):
            group=[r for r in rows if r["topology"]==topology and r["task"]==task]
            changed=any(abs(r["selected_cutoff_Hz"]-r["current_cutoff_Hz"])>1e-6 for r in group)
            if not changed:
                figures.append(dict(topology=topology,task=task,reused_unchanged=True,
                                    png=str(study/(topology+"-"+task.lower()+"-sampling-aware.png"))))
                continue
            fig,axes=plt.subplots(4,2,figsize=(13,13.7),squeeze=False)
            for ax,row in zip(axes.flat,group):panel(ax,row)
            fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=4,frameon=False)
            fig.tight_layout(rect=(0,0,1,.96),h_pad=1.6)
            path=out/(topology+"-"+task.lower()+"-selected.png")
            for ext in("png","svg"):fig.savefig(path.with_suffix("."+ext),dpi=180)
            plt.close(fig);figures.append(dict(topology=topology,task=task,reused_unchanged=False,png=str(path)))
    keys=[("kagome","periodic",0),("kagome","full",383),("hex","periodic",0),("hex","full",336)]
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.2),squeeze=False)
    for ax,key in zip(axes.flat,keys):panel(ax,next(r for r in rows if(r["topology"],r["design"],r["id"])==key and r["task"]=="FT"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in("png","svg"):fig.savefig(out/("cutoff-examples."+ext),dpi=200)
    plt.close(fig)
    return figures


def run(root=ROOT):
    root=Path(root); out=root/"audit/curve-processing/accepted-filter-study";out.mkdir(parents=True,exist_ok=True)
    registry_path=root/"audit/curve-processing/periodic-cutoffs.json"
    registry={(r["topology"],r["task"]):r for r in json.loads(registry_path.read_text())["families"]}
    rows=[]; arrays={}
    for row in records(root):
        path=Path(row["native"])
        if sha(path)!=row["native_sha256"]:raise ValueError("Changed native source: "+str(path))
        with np.load(path,allow_pickle=False)as d:t,x,y=[d[k] for k in("time","x","y")]
        control=registry[row["topology"],row["task"]]
        (fx,fy),info=archive_filter(t,x,y,control["natural_frequency_Hz"])
        key=row["topology"],row["task"],row["design"],row["id"]
        arrays[key]=(t,x,y,fx,fy)
        before,after=numerical_metrics(x,y),numerical_metrics(fx,fy)
        item={k:row[k] for k in("topology","task","design","id")}
        item.update(native=str(path),native_sha256=sha(path),samples=len(t),filter=info,raw=before,filtered=after,
                    work_change_percent=100*(after["full_work"]/before["full_work"]-1)if before["full_work"] else None,
                    roughness_reduction_percent=100*(1-after["roughness"]/before["roughness"])if before["roughness"]else None,
                    stiffness=qualify_stiffness(t,x,y,control["natural_frequency_Hz"]))
        rows.append(item)
    gold=[]
    source_records=json.loads((root/"audit/curve-processing/periodic-cutoff-native/index.json").read_text())
    for c in json.loads((root/"audit/controls/uat-initial-response/uat-linear-summary.json").read_text()):
        row=next(r for r in source_records if r["task"]=="UT"and r["source_inp"].lower()==c["source_inp"].lower())
        with np.load(row["native"],allow_pickle=False)as d:t,x,y=[d[k]for k in("time","x","y")]
        fit=qualify_stiffness(t,x,y,registry[row["topology"],"UT"]["natural_frequency_Hz"])
        gold.append(dict(job=c["job"],gold_E_MPa=c["E_apparent_MPa"],curve_estimate_MPa=fit["estimate"],
                         relative_error_percent=100*(fit["estimate"]/c["E_apparent_MPa"]-1),qualification=fit))
    ft_gold=ft_gold_controls(root)
    report=dict(policy="Qualified archive adaptation; no source/population promotion in this module", archive_cutoff="2fn when below saved Nyquist; otherwise fs/6",
        order=2,phase="causal SOS/DC; paired coordinates and responses",primary_source=SOURCE,
        no_final_figures_or_source_data_modified=True,records=rows,gold_controls=gold,FT_gold_controls=ft_gold,
        limitations=["No postprocessing filter can undo aliasing already present","A numerical stability screen alone does not prove an infinitesimal elastic tangent",
                    "Initial work-equivalent stiffness is not a full homogenised tensor"],
        periodic_registry_sha256=sha(registry_path))
    (out/"qualification.json").write_text(json.dumps(report,indent=2)+"\n")
    summary=[{k:v for k,v in row.items()if k in("topology","task","design","id","samples","work_change_percent","roughness_reduction_percent")}
             | {"stiffness_estimate":row["stiffness"]["estimate"],"stiffness_quality":row["stiffness"]["quality"],
                "window_relative_spread":row["stiffness"]["window_relative_spread"],"cutoff_relative_spread":row["stiffness"]["cutoff_relative_spread"]}for row in rows]
    with(out/"metrics.csv").open("w",newline="")as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    plot(out,rows,arrays,gold)
    print(json.dumps(dict(histories=len(rows),gold=[{k:v for k,v in r.items()if k!="qualification"}for r in gold],
        quality_counts={task:sum(r["task"]==task and r["stiffness"]["quality"]=="passes_frozen_numerical_screens"for r in rows)for task in("UT","FT")}),indent=2))
    return report


def plot(out,rows,arrays,gold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    for topology in COLORS:
        for task in("UT","FT"):
            group=[r for r in rows if r["topology"]==topology and r["task"]==task]
            fig,axes=plt.subplots(4,2,figsize=(13,13.7),squeeze=False)
            for ax,row in zip(axes.flat,group):
                t,x,y,fx,fy=arrays[topology,task,row["design"],row["id"]]
                scale=1. if task=="UT"else.001
                ax.plot(x,y*scale,color=".7",lw=.8)
                ax.plot(x,np.asarray(smooth(smooth(y)))*scale,color="#775599",ls="--",lw=1.2)
                ax.plot(fx,fy*scale,color=COLORS[topology],lw=1.25)
                name="Periodic"if row["design"]=="periodic"else("Full disorder"if row["design"]=="full"else"Mechanism-based")+", ID"+str(row["id"])
                ax.set_title(name,color={"periodic":".1","full":"#ff7f0e","mechanism":"#1f77b4"}[row["design"]],fontsize=10)
                ax.text(.98,.04,"fc={:.1f}Hz; periodic2fn={:.1f}Hz".format(row["filter"]["cutoff_Hz"],row["filter"]["physical_cutoff_Hz"]),
                        transform=ax.transAxes,ha="right",fontsize=8,bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
                ax.set_xlabel(r"Macroscopic strain, $\varepsilon_L$"if task=="UT"else"Load-line displacement (mm)")
                ax.set_ylabel(r"Macroscopic stress, $\Sigma$ (MPa)"if task=="UT"else"Force (kN)")
                ax.grid(alpha=.18)
            fig.suptitle(topology+" | "+task+" | sampling-aware archive filtering",y=.995)
            fig.legend(handles=[Line2D([],[],color=".7",label="Raw"),Line2D([],[],color=COLORS[topology],label=r"Causal Butterworth, resolved $2f_n$ or $f_s/6$ fallback"),
                                Line2D([],[],color="#775599",ls="--",label="Legacy double moving average")],loc="upper center",bbox_to_anchor=(.5,.977),ncol=3,frameon=False)
            fig.tight_layout(rect=(0,0,1,.944),h_pad=1.6)
            for suffix in("png","svg"):fig.savefig(out/(topology+"-"+task.lower()+"-sampling-aware."+suffix),dpi=180)
            plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    axes[0].bar(np.arange(len(gold)),[g["relative_error_percent"]for g in gold],color=["#008000","#008000","black","black"])
    axes[0].axhline(0,color=".2",lw=.8);axes[0].set_xticks(np.arange(len(gold)),[g["job"].removeprefix("uat_").replace("_","\n")for g in gold])
    axes[0].set_ylabel("Curve estimate minus elastic-control modulus (%)")
    for task,color in(("UT","#1f77b4"),("FT","#ff7f0e")):
        group=[r for r in rows if r["task"]==task]
        axes[1].scatter([r["stiffness"].get("window_relative_spread",np.nan)or 0 for r in group],
                        [r["stiffness"].get("cutoff_relative_spread",np.nan)or 0 for r in group],label=task,color=color,s=18)
    axes[1].axvline(.05,ls="--",color=".5");axes[1].axhline(.03,ls="--",color=".5")
    axes[1].set_xlabel("30/40/50% window sensitivity");axes[1].set_ylabel("Sampling cutoff sensitivity");axes[1].legend()
    fig.tight_layout()
    for suffix in("png","svg"):fig.savefig(out/("stiffness-qualification."+suffix),dpi=210)
    plt.close(fig)


def flagged_curve_summary(record,prediction,control,cutoff_Hz):
    """Describe the frozen curve flag independently of the later FE error."""
    key=record["design"],record["id"]
    if key!=(prediction["design"],prediction["id"]) or key!=(control["design"],control["id"]):
        raise ValueError("Flagged curve and holdout identities differ")
    fit=record["stiffness"]
    if fit["estimate"]!=prediction["frozen_curve_estimate_MPa"]:
        raise ValueError("Frozen curve prediction differs")
    if fit["quality"]!="review_required" or fit["flags"]!=["window_sensitive_over5percent"]:
        raise ValueError("Expected the original window-only preflag")
    return dict(design=key[0],id=key[1],reason="30/40/50% window spread exceeds preselected5% screen",
                window_spread_percent=100*fit["window_relative_spread"],
                cutoff_spread_percent=100*fit["cutoff_relative_spread"],
                energy_curvature_percent=100*fit["energy_fit_residual"],
                selected_cutoff_Hz=float(cutoff_Hz),E30_MPa=fit["windows"]["0.3"]["estimate"],
                E40_MPa=fit["estimate"],E50_MPa=fit["windows"]["0.5"]["estimate"],
                samples40=fit["samples"],elastic_control_MPa=control["elastic_control_E_apparent_MPa"],
                curve_vs_FE_error_percent=control["relative_error_percent"],
                curve_vs_FE_within5percent=control["within_five_percent"],
                flag_means_missing_value_or_failed_simulation=False,
                raw_transfer=record["path"],raw_transfer_sha256=record["source_sha256"])


def flagged_hex_curves(root=ROOT):
    """Read and visualise all9 frozen flags; never alter their accepted catalogue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from validation_curve_processing import transfer_prefix
    root=Path(root);study=root/"audit/curve-processing/accepted-filter-study"
    out=study/"flagged-hex-curves";out.mkdir(parents=True,exist_ok=True)
    frozen_path=study/"holdout-controls/frozen-predictions.json"
    population_path=study/"hex-population-readonly-qualification.json"
    comparison_path=study/"holdout-controls/holdout-comparison.json"
    catalogue=study/"accepted-hex-stiffness.json"
    catalogue_csv=study/"accepted-hex-stiffness.csv"
    protected=[frozen_path,population_path,comparison_path,catalogue,catalogue_csv]
    before={str(path):sha(path) for path in protected}
    freeze=json.loads(frozen_path.read_text());population=json.loads(population_path.read_text())
    if sha(population_path)!=freeze["source_replay_sha256"]:
        raise ValueError("Frozen population replay changed")
    records={(r["design"],r["id"]):r for r in population["records"]}
    comparison={(r["design"],r["id"]):r for r in json.loads(comparison_path.read_text())["records"]}
    accepted={(r["design"],r["id"]):r for r in json.loads(catalogue.read_text())["records"]}
    registry_path=root/"audit/curve-processing/periodic-cutoffs.json"
    registry=json.loads(registry_path.read_text())
    fn=next(r["natural_frequency_Hz"] for r in registry["families"] if r["topology"]=="hex" and r["task"]=="UT")
    rows,curves=[],[]
    if len(freeze["predictions"])!=9:
        raise ValueError("Exactly nine frozen flagged specimens required")
    for prediction in freeze["predictions"]:
        key=prediction["design"],prediction["id"];record=records[key]
        if sha(record["path"])!=record["source_sha256"] or sha(prediction["source_inp"])!=prediction["source_inp_sha256"]:
            raise ValueError("Frozen flagged raw transfer/source INP changed")
        if accepted[key]["estimator"]!="source_exact_linear_UAT_preflagged_fallback":
            raise ValueError("Accepted preflagged fallback membership changed")
        raw,support=transfer_prefix(record["path"])
        time=np.arange(len(raw))*.0005
        (x,y),info=final_archive_filter(time,raw[:,0],raw[:,1],fn)
        fits={str(f):initial_work_stiffness(x,y,f) for f in(.3,.4,.5)}
        for fraction,fit in fits.items():
            if not np.isclose(fit["estimate"],record["stiffness"]["windows"][fraction]["estimate"],rtol=1e-12,atol=1e-10):
                raise ValueError("Selected filter/window no longer reproduces frozen estimate")
        row=flagged_curve_summary(record,prediction,comparison[key],info["cutoff_Hz"])
        stem="hex-UT-"+key[0]+"-"+str(key[1])
        csv_path=out/(stem+"-inspection.csv")
        with csv_path.open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["original_sample_index","declared_time_s","raw_macro_strain","raw_macro_stress_MPa","filtered_macro_strain","filtered_macro_stress_MPa","selected40percent_work_line_MPa","within30percent_window","within40percent_window","within50percent_window"])
            E=fits["0.4"]["estimate"]
            writer.writerows((i,float(t),float(a),float(b),float(c),float(d),float(E*(c-x[0])),
                              i<=fits["0.3"]["end_index"],i<=fits["0.4"]["end_index"],i<=fits["0.5"]["end_index"])
                             for i,(t,a,b,c,d) in enumerate(zip(time,raw[:,0],raw[:,1],x,y)))
        row.update(source_inp=prediction["source_inp"],source_inp_sha256=prediction["source_inp_sha256"],
                   accepted_E_MPa=accepted[key]["E_apparent_MPa"],inspection_csv=str(csv_path),
                   samples=len(raw),support=support,filter=info,
                   window_end_indices={k:v["end_index"] for k,v in fits.items()})
        rows.append(row);curves.append((raw,x,y,fits))
    figures=[]
    for zoom in(False,True):
        fig,axes=plt.subplots(3,3,figsize=(15,11.7))
        for ax,row,(raw,x,y,fits) in zip(axes.flat,rows,curves):
            color="#ff7f0e" if row["design"]=="full" else "#1f77b4"
            end50=fits["0.5"]["end_index"]
            ax.plot(raw[:,0],raw[:,1],color=".65",lw=.9,zorder=1)
            ax.plot(x,y,color=color,lw=1.4,zorder=3)
            end=fits["0.4"]["end_index"];line_x=np.array([x[0],x[end]])
            ax.plot(line_x,fits["0.4"]["estimate"]*(line_x-x[0]),color=".1",ls="--",lw=1.5,zorder=4)
            for fraction,marker in(("0.3","^"),("0.4","o"),("0.5","s")):
                index=fits[fraction]["end_index"]
                ax.scatter(x[index],y[index],s=32,marker=marker,facecolor="white",edgecolor=".1",zorder=5)
            if zoom:
                xmax=max(float(x[end50]),float(raw[min(end50,len(raw)-1),0]))*1.2
                local=(raw[:,0]<=xmax)
                ymax=max(float(np.max(raw[local,1])),float(np.max(y[:end50+1])))
                ax.set_xlim(-.02*xmax,xmax);ax.set_ylim(-.04*ymax,1.18*ymax)
                for fraction in("0.3","0.4","0.5"):
                    index=fits[fraction]["end_index"]
                    ax.annotate(str(int(float(fraction)*100))+"%",(x[index],y[index]),xytext=(3,5),textcoords="offset points",fontsize=8)
            title=("Full disorder" if row["design"]=="full" else "Mechanism-based")+" | ID"+str(row["id"])
            ax.set_title(title,color=color,fontsize=10)
            note="Window {:.2f}% (>5%); cutoff {:.2f}%; curvature {:.2f}%\nCurve−elastic control: {:+.2f}% {}".format(
                row["window_spread_percent"],row["cutoff_spread_percent"],row["energy_curvature_percent"],
                row["curve_vs_FE_error_percent"],"(within5%)" if row["curve_vs_FE_within5percent"] else "(outside5%)")
            ax.text(.02,.98,note,ha="left",va="top",transform=ax.transAxes,fontsize=8,
                    bbox=dict(facecolor="white",alpha=.85,edgecolor="none",pad=2))
            ax.grid(alpha=.17);ax.set_xlabel(r"Macroscopic strain, $\varepsilon_L$",fontsize=9)
            ax.set_ylabel(r"Macroscopic stress, $\Sigma$ (MPa)",fontsize=9)
        handles=[Line2D([],[],color=".65",lw=1,label="Raw recorded transfer"),
                 Line2D([],[],color="#1f77b4",lw=1.5,label=r"Selected causal Butterworth, $f_c=\min(2f_n,f_s/6)$"),
                 Line2D([],[],color=".1",ls="--",label="40%-peak work-equivalent line"),
                 Line2D([],[],color=".1",marker="^",mfc="white",ls="none",label="30% check"),
                 Line2D([],[],color=".1",marker="s",mfc="white",ls="none",label="50% check")]
        fig.suptitle("Nine preflagged hexagonal UAT specimens | "+("initial loading" if zoom else "full recorded curves"),y=.997,fontsize=14)
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.975),ncol=3,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.925),h_pad=1.9,w_pad=1.5)
        stem="flagged-hex-initial-loading" if zoom else "flagged-hex-full-curves"
        for extension in("png","svg"):
            path=out/(stem+"."+extension);fig.savefig(path,dpi=210);figures.append(str(path))
        plt.close(fig)
    for path in protected:
        if sha(path)!=before[str(path)]:raise ValueError("Protected source/catalogue changed")
    fields=[k for k in rows[0] if k not in("support","filter","window_end_indices")]
    with(out/"flagged-specimens.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields}for r in rows)
    report=dict(records=rows,figures=figures,flag_count=9,reason="All9 window sensitivities exceed5%; none is a missing stiffness or solver-failure flag",
                within5percent_FE=sum(r["curve_vs_FE_within5percent"] for r in rows),
                outside5percent_FE=sum(not r["curve_vs_FE_within5percent"] for r in rows),
                selected_filter="min(2fn,fs/6), causal paired second-order/DC; all9 cutoffs333.333Hz",
                timestamp_basis="Declared0.0005s transfer schedule; transfer files do not store native solver timestamps",
                numerical_core_sha256=numerical_core_sha256(),protected_hashes_before=before,
                protected_hashes_after={str(path):sha(path)for path in protected},
                no_source_catalogue_Props_or_MLdata_writes=True,new_solver_jobs=0)
    (out/"flagged-curves.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main(argv=None):
    """Default to the selected single policy; old qualification is reference-only."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--action",choices=("selected","accepted-hex","uniform-study","flagged-hex-curves"),default="selected")
    args=parser.parse_args(argv)
    if args.action=="flagged-hex-curves":
        report=flagged_hex_curves(args.root);summary={k:report[k]for k in("flag_count","within5percent_FE","outside5percent_FE")}
    elif args.action=="accepted-hex":
        report=accepted_hex_stiffness(args.root);summary=report["counts"]
    elif args.action=="uniform-study":
        report=compare_uniform_fs6(args.root);summary={k:report[k] for k in("native_histories","changed_histories")}
    else:
        report=compare_capped_policy(args.root)
        summary={k:report[k] for k in("selected_policy","native_histories","changed_histories","stiffness_windows_passed")}
    print(json.dumps(summary,indent=2))
    return report


if __name__=="__main__":
    main()

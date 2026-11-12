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
            receipt_path=out/"reviewed-code-update.json"
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            expected_receipt=dict(frozen_predictions_sha256=sha(frozen_path),
                original_estimator_sha256=previous["estimator_sha256"],
                reviewed_estimator_sha256=freeze["estimator_sha256"],
                numerical_core_sha256=numerical_core_sha256())
            if any(receipt.get(k)!=v for k,v in expected_receipt.items()):
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


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=ROOT)
    run(parser.parse_args().root)

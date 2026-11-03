"""Read completed CNN2 context-removal fits; preserve the accepted winner."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
from resources.MLfeature_ablation import CASES,GROUPS
from resources.MLstudy_reports import (completed_field_review,verify_study_inventory,
    replace_study_followup,serial_schematic)


def verify_feature_mask(config,names,mode,variant):
    """Check exactly the declared task/group is suppressed, including reload data."""
    stage=config["model_config"]["field_model"]
    if not stage.get("independent_tasks"):raise ValueError("Require independent field models")
    if stage.get("locality",{}).get("kind")!="cnn" or stage["locality"].get("blocks")!=2:
        raise ValueError("Feature screen is not CNN2")
    keep=stage.get("context_keep")
    if variant=="control":
        if keep is not None:raise ValueError("Control unexpectedly masks context")
        return []
    removed=list(names) if variant=="no_context" else list(GROUPS[variant])
    expected={m:[name not in removed if m==mode else True for name in names] for m in ("UT","FT")}
    if keep!=expected:raise ValueError("Saved context mask differs from requested attribution")
    if config["model_config"]["curve_model"].get("context_keep") is not None:
        raise ValueError("Field screen unexpectedly changes curve context")
    return removed


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/input-feature-audit")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    a=p.parse_args(argv);out=a.out;root=a.root/"MULTI/Dual/Transformer"
    receipt=verify_study_inventory(a.root,out/"download-inventory.json",15)
    (out/"download-verification.json").write_text(json.dumps(receipt,indent=2))
    cases=[];contract={};masks=[]
    for mode,variant in CASES:
        path=root/f"nodal-features-261102-{variant}-{mode}-s42"
        data=json.loads((path/"model_data.json").read_text());cfg=json.loads((path/"model.json").read_text())
        removed=verify_feature_mask(cfg,data["task_feature_names"],mode,variant)
        if cfg["training"].get("field_only_modes")!=[mode]:raise ValueError("Feature fit trains the wrong task")
        for key,value in (("normalizers",data["normalizers"]),("context",data["reference_task_features"]),("training",{k:v for k,v in cfg["training"].items() if k not in ("best_epoch","best_loss","field_only_modes")}),
            ("architecture",{k:v for k,v in cfg["model_config"]["field_model"].items() if k!="context_keep"})):
            digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            if contract.setdefault(key,digest)!=digest:raise ValueError("Feature comparison changed "+key)
        masks.append(dict(mode=mode,variant=variant,removed=",".join(removed),path=str(path)))
        cases.append(dict(mode=mode,seed=42,variant=variant,path=path))
    api=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    fields,regions,text=completed_field_review(cases,a.data_root,api,out,"results","Completed nodal feature attribution",
        "All15 final fits completed: UT6 and FT9, each independently trained with CNN2 and masked MSE. Only specified physical-context channels are suppressed; input width11, parameters, common initialization, geometry, masks, normalization and training budget are preserved. Pin suppression occurs after per-specimen refresh. Operational FT attention/loss absence masks remain active, even in no_present/no_context arms. One seed42,793 identical validation specimens; no curve training or locked-test evaluation.",
        "CNN2/MSE unchanged; suppress selected\ncontext columns before their projection",
        "**Conditional contribution screen, not a feature promotion.** A small one-seed change does not establish a removable feature. Reference coordinates, positional encoding, CNN grid and tip channels overlap; a removal result is conditional on the retained representation.")
    serial_schematic(out,"results-architecture.png",[
        ("Full-context CNN2/MSE control",["Disorder +11\ncontext channels","2 masked CNN blocks\n→ field Transformer","20-frame\nU1/U2 fields"],"All context active; operational FT absence mask retained"),
        ("One-group removal",["Same inputs; mask\nselected context","Same CNN2 +\nfield Transformer","Same20-frame\nU1/U2 fields"],"Saved mask after pin refresh; same width/MSE/initialization")])
    summary=fields[["mode","variant","change_percent","skill_percent","hours"]].copy()
    for region,label in (("STATUS onset endpoints ±1 frame","endpoint_change_percent"),("STATUS onset strut relative motion ±1 frame","strut_change_percent")):
        values=regions[regions.region==region].set_index(["mode","variant"]).change_percent
        summary[label]=[values.loc[r.mode,r.variant] for r in summary.itertuples()]
    summary.to_csv(out/"decision-metrics.csv",index=False);pd.DataFrame(masks).to_csv(out/"suppressed-channels.csv",index=False)
    lines=[]
    for mode in ("UT","FT"):
        part=summary[summary["mode"]==mode];best=part.sort_values("change_percent").iloc[0]
        lines.append(f"**{mode}:** lowest global error is {best.variant} ({best.change_percent:+.2f}% versus full-feature CNN2); confirmed endpoints {best.endpoint_change_percent:+.2f}%, signed failing-strut motion {best.strut_change_percent:+.2f}%. One seed is screening evidence; do not combine separate removals and assume their gains add.")
    lines.append("**Decision:** retain UT's current context: every removal worsens global error, though some improve signed strut motion. FT no_xy is the strongest removal candidate (global −2.15%, endpoints −2.09%, strut motion −2.15%); no_pins also helps, but takes 5.56h versus 4.35h for full context. Do not remove both together based on separate fits. Confirm no_xy on the chosen FT depth with matched seeds before changing the feature contract. Retaining all FT context is the conservative interim choice; removing all context worsens both global and fracture errors. These CNN2 results do not establish the same attribution on CNN3 or in the curve stage.")
    text=text.replace("### Global performance","\n\n".join(lines)+"\n\n### Global performance",1)
    text+="\n### Exact feature intervention and limitations\n\nReference x0/y0 removal does not remove the fixed CNN grid or node positional encoding. No-context retains disorder, those positions and task-vector bias. Raw present is constant1 at participating nodes and is distinct from the mandatory mask. The pin flags are sample-specific coupling membership; they are not individually prescribed displacements. All tip channels are reference-geometry descriptors, not future fracture labels. The saved model-side context_keep mapping is checked against every declared arm; physical data/scalers/context values stay unchanged. [Exact suppressed channels](suppressed-channels.csv) · [Decision metrics](decision-metrics.csv) · [Download verification](download-verification.json). The original feature rationale below is retained as interpretation, not prior training evidence.\n"
    replace_study_followup(out/"README.md","feature-results",text)
    print(summary.to_string(index=False));print("REPORT",out/"README.md")


if __name__=="__main__":main()

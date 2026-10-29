"""Review completed MAE/Smooth-L1 or FT-box spatial fits; no training."""
import argparse
import json
from pathlib import Path
import runpy
import pandas as pd
from resources.MLstudy_reports import (completed_field_review,verify_study_inventory,figure)


def study_verdict(summary,study):
    if study=="robust-fields":
        return ("**Verdict: retain MSE.** Every tested alternative worsens global error in both tasks. UT fracture-endpoint errors also worsen for every alternative. FT Smooth-L1 beta1 has a small signed failing-strut improvement with slightly worse global error; that is a one-seed trade-off, not enough to replace MSE. MAE worsens both global and confirmed-fracture errors, so do not adopt it at this setting. A plausible explanation is that reducing large-residual influence underemphasizes the rare mechanically important events; these results do not prove that mechanism or reject every robust-loss threshold. No coefficient/architecture combinations were tested.\n\n")
    small=summary[summary.variant=="0.001"].iloc[0]
    return (f"**Verdict: FT-box spatial0.001 is a promising one-seed candidate:** global RMSE {small.global_change_percent:+.2f}%, "
        f"confirmed endpoints {small.onset_endpoint_change_percent:+.2f}% and signed failing-strut motion {small.failing_strut_change_percent:+.2f}%. "
        "Restricting the term avoids the severe whole-field deterioration seen at the larger all-edge coefficient, but does not make every regional coefficient beneficial. At0.01 the narrower events improve while the broader crack-tip field worsens;0.1 remains unsuitable. Confirm the small coefficient before adoption, and keep it separate from the confirmed CNN2 effect. No CNN+spatial combination or downstream curve gain is established here.\n\n")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("study",choices=("robust-fields","spatial-box"))
    p.add_argument("--root",type=Path,default=Path("Z:/p2"))
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    report_folder={"robust-fields":"robust-field-loss","spatial-box":"ft-box-spatial"}[a.study]
    out=a.out or Path(__file__).resolve().parents[3]/"samples/loss-study"/report_folder
    root=a.root/"MULTI/Dual/Transformer";out.mkdir(parents=True,exist_ok=True)
    expected=8 if a.study=="robust-fields" else 4
    receipt=verify_study_inventory(a.root,out/"download-inventory.json",expected)
    (out/"download-verification.json").write_text(json.dumps(receipt,indent=2))
    cases=[]
    if a.study=="robust-fields":
        for mode in ("UT","FT"):
            for variant in ("control","mae","smooth_l1_0p1","smooth_l1_1"):
                cases.append(dict(mode=mode,seed=42,variant=variant,path=root/f"robust-fields-261028-{variant}-{mode}-s42"))
        title="Independent field MSE, MAE and Smooth-L1"
        question="Eight final field fits isolate the pointwise objective: MSE, MAE, Smooth-L1 beta0.1 and beta1, independently for UT/FT. Architecture384/8/3, inputs, masks, train-fitted normalization, optimization budget and physical validation-MSE selection remain fixed. One seed42; no CNN, derivative, STATUS auxiliary loss or downstream curves."
        change="MSE → MAE or Smooth-L1;\nall other settings unchanged"
        detail="MAE gives large residuals less influence than MSE. Smooth-L1 is quadratic below beta and linear above it. Beta is in standardized field coordinates, not millimetres; these beta choices also change the objective scale. This is a screening test, not a tuned optimum. Fixed physical errors, rather than unlike objective totals, determine the comparison."
    else:
        cases.append(dict(mode="FT",seed=42,variant="control",source_variant="spatial_0p0",path=root/"spatial-balance-261028-spatial-0-FT-s42"))
        for coefficient,token in ((.001,"0p001"),(.003,"0p003"),(.01,"0p01"),(.1,"0p1")):
            cases.append(dict(mode="FT",seed=42,variant=str(coefficient),source_variant="box_spatial_"+str(coefficient).replace(".","p"),
                path=root/f"spatial-box-261028-box-spatial-{token}-FT-s42"))
        title="FT crack-tip-box-only spatial supervision"
        question="Four FT final fits restrict only the extra signed-neighbour-motion loss to struts whose two initial endpoints lie inside the fixed geometric crack-tip box. Coefficients0.001/0.003/0.01/0.1 reuse the freshly trained zero control from the all-edge balance screen. Whole-field displacement MSE remains active everywhere. No STATUS labels select training regions; one seed42, no CNN or curves."
        change="Whole-field MSE + spatial term\nonly on pairs inside FT tip box"
        detail="Only the added spatial penalty is regional. Global attention and shared field parameters still permit effects outside the box. Both endpoints must be valid and inside the box; coordination-balanced averaging and TRAIN-only scales remain unchanged. This is different from weighting displacement MSE inside the box."
    api=runpy.run_path(str(Path(__file__).with_name("review_fracture_regions.py")))
    fields,regions,text=completed_field_review(cases,a.data_root,api,out,"review",title,question,change,
        "**Completed, checksum-verified validation screen. No replacement has been accepted.** Global and confirmed-fracture effects must be considered together; one-seed gains need confirmation before adoption.")
    summaries=[]
    for mode in fields["mode"].drop_duplicates():
        for row in fields[(fields["mode"]==mode)&(fields.variant!="control")].itertuples():
            r=regions[(regions["mode"]==mode)&(regions.variant==row.variant)].set_index("region")
            summaries.append(dict(mode=mode,variant=row.variant,global_change_percent=row.change_percent,
                onset_endpoint_change_percent=r.loc["STATUS onset endpoints ±1 frame","change_percent"],
                failing_strut_change_percent=r.loc["STATUS onset strut relative motion ±1 frame","change_percent"],
                skill_vs_train_mean_percent=row.skill_percent))
    summary=pd.DataFrame(summaries);summary.to_csv(out/"decision-metrics.csv",index=False)
    intro="\n\n".join(f"**{r.mode} {r.variant}:** global RMSE {r.global_change_percent:+.2f}%; confirmed fracture endpoints {r.onset_endpoint_change_percent:+.2f}%; signed failing-strut motion {r.failing_strut_change_percent:+.2f}%." for r in summary.itertuples())
    intro=study_verdict(summary,a.study)+intro
    text=text.replace(f"## {title}","## Recap, findings and verdict",1)
    text=text.replace("### Global performance",intro+"\n\n### Global performance",1)
    if a.study=="spatial-box":
        all_edges=pd.read_csv(out.parent/"spatial-balance-metrics.csv");all_regions=pd.read_csv(out.parent/"spatial-balance-regions.csv")
        comparison=pd.concat([all_regions.assign(scope="All initial edges"),regions.assign(scope="FT box edges only")],ignore_index=True)
        comparison.to_csv(out/"all-edge-versus-box.csv",index=False)
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(19,6))
        for ax,region in zip(axes,("Whole field","STATUS onset endpoints ±1 frame","FT crack-tip box")):
            for scope,group in comparison[comparison.region==region].groupby("scope",sort=False):
                values=group[group.variant!="control"].copy();values["coefficient"]=values.variant.astype(float);values=values.sort_values("coefficient")
                ax.plot(values.coefficient,values.change_percent,"o-",label=scope)
            ax.set(xscale="log",xlabel="Spatial coefficient",ylabel="RMSE change vs same zero control (%)",title=region)
            ax.axhline(0,color="black",lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=10)
            if comparison.change_percent.abs().max()>30:ax.set_yscale("symlog",linthresh=5)
        fig.tight_layout();fig.savefig(out/"regional-versus-all-edge.png",dpi=170);plt.close(fig)
        text+="\n### Does restricting the term avoid broader deterioration?\n\nSame zero control and coefficient; only the spatial edge region differs. Symlog axes retain severe deterioration while showing small changes near zero.\n\n"+figure(out,"regional-versus-all-edge.png","All-edge versus box-only spatial loss on global, fracture endpoint and broader box errors")
    text+="\n### Exact intervention\n\n"+detail+"\n\n[Decision metrics](decision-metrics.csv) · [Download verification](download-verification.json). No default or loss implementation changed by this review.\n"
    (out/"README.md").write_text("# "+title+"\n\n"+text,encoding="utf-8")
    print(summary.to_string(index=False));print("REPORT",out/"README.md")


if __name__=="__main__":main()

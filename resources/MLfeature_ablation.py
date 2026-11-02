"""Opt-in physical-context ablations; preserve all geometry/validity masks."""
from resources.MLlocality import configure_field_locality

GROUPS={
    "no_xy":("x0","y0"),"no_designable":("designable",),
    "no_present":("present",),"no_interfaces":("bottom_interface","top_interface"),
    "no_pins":("coupled_fixity","coupled_load"),
    "no_tip":("tip_dx_ref","tip_dy_ref","tip_distance_ref"),
    "no_tip_offsets":("tip_dx_ref","tip_dy_ref"),"no_tip_distance":("tip_distance_ref",),
}
CASES=tuple((mode,variant) for mode,variants in (
    ("UT",("control","no_xy","no_designable","no_interfaces","no_present","no_context")),
    ("FT",("control","no_xy","no_designable","no_pins","no_tip","no_tip_offsets","no_tip_distance","no_present","no_context"))) for variant in variants)


def configure_context_ablation(network,data,args):
    """CNN2/MSE baseline; one input group suppressed, not physical site removal."""
    if (args.mode,args.variant) not in CASES:raise ValueError("Unsupported task/feature ablation")
    configure_field_locality(network,data,"cnn2")
    if args.variant=="control":return
    names=list(data.task_feature_names)
    removed=names if args.variant=="no_context" else GROUPS[args.variant]
    if any(name not in names for name in removed):raise ValueError("Feature schema differs from study")
    keep={m:[True]*len(names) for m in ("UT","FT")}
    for name in removed:keep[args.mode][names.index(name)]=False
    for stage in network.field_model.tasks.values():stage.set_context_keep(keep)

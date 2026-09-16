#!/usr/bin/env python3
"""Controlled field-loss ablation using the archived independent HPO winner."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--task", choices=["UT", "FT"], required=True)
    parser.add_argument("--hpo-model-json")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    source = Path(args.hpo_model_json or
        f"/data/SEMS-TaoLab/Niccolo-Forte/p2/{args.task}/Field/HPO/f{args.task}-fHPO/Transformer/best_model.json")
    cfg = json.loads(source.read_text())["reload_config"]["training_config"]
    scheduler, early = cfg["scheduler"], cfg["earlyStop"]["params"]
    preset = ["--task",args.task,"--hpo-model-json",str(source),"--epochs","450",
              "--field-loss-variant","baseline","--eval-split","val","--selection-metric","mse",
              "--batch",str(cfg["batch"]),"--lr",str(cfg["lr"]),"--weight-decay",str(cfg["opt"][1]),
              "--scheduler-factor",str(scheduler[2]),"--scheduler-patience",str(scheduler[3]),
              "--scheduler-threshold",str(scheduler[4]),"--early-stop-patience",str(early["patience"]),
              "--early-stop-delta",str(early["min_delta"])]
    curve_source=source.parents[5]/args.task/'FieldToCurve'/'HPO'/f'f2c{args.task}full-fHPO'/'Transformer'/'best_model.json'
    if curve_source.is_file():preset += ['--curve-model-json',str(curve_source)]
    else:print(f'No adjacent frozen curve checkpoint at {curve_source}; supply --curve-model-json for the bridge comparison.')
    return runpy.run_path(str(Path(__file__).with_name("A0-HPC_Field-test.py")))["main"](preset+rest)


if __name__ == "__main__":
    main()

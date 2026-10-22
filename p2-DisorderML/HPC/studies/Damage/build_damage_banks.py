"""Build aligned fracture banks in explicit staging; no archive publishing."""
import argparse
import json

from resources.damage_data import build_damage_banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mldata-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Explicit staging destination; no archive publishing")
    parser.add_argument("--transfer-root", action="append", default=[], help="Explicit periodic/disordered transfer root")
    parser.add_argument("--inventory", help="fcc_status_batch_v1 manifest containing exact FIELDd output paths")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reports = build_damage_banks(args.mldata_dir, args.output_dir, args.transfer_root, args.inventory, args.overwrite)
    print(json.dumps([{k: report[k] for k in ("output", "npz_sha256", "samples", "frames", "struts",
                                            "unknown_parent_frames", "failed_parent_frames")} for report in reports], indent=2))


if __name__ == "__main__":
    main()

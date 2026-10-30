"""Reproduce C(T) modal controls from exact indexed validation inputs.

Prepare never edits a source input. Run is an explicit, local Abaqus/Standard
linear-perturbation submission. Extract must run in Abaqus Python. The released
control retains the fixed pin U1/U2 and loading-pin U1, releasing loading U2;
the held control additionally fixes loading U2 at zero actuator displacement.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


PRIMARY_INPUTS = (
    "10/0.2/Fracture-tri-70-per-1.inp",
    "10/0.2/Fracture-kagome-40-per-1.inp",
    "10/0.2/Fracture-hex-60-per-1.inp",
    "3/0.16/Fracture-hex-80-per-1.inp",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks(text):
    """Yield complete Abaqus keyword blocks, retaining comments with their block."""
    found = list(re.finditer(r"(?m)^\*(?!\*)[^\r\n]*", text))
    return [text[m.start():found[i + 1].start() if i + 1 < len(found) else len(text)]
            for i, m in enumerate(found)]


def keyword(block):
    return block.splitlines()[0].split(",", 1)[0].strip().lower()


def numeric_line(block):
    return next(line for line in block.splitlines()[1:]
                if line.strip() and not line.startswith("**"))


def job_directory(out, name):
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise ValueError("Unsafe generated job name: " + name)
    directory = (Path(out).resolve() / name).resolve()
    if directory.parent != Path(out).resolve():
        raise ValueError("Job directory escapes the selected output root")
    return directory


def prepare(args):
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    case_index = getattr(args, "case_index", None) or root / "case-index.csv"
    with Path(case_index).open(newline="", encoding="utf-8-sig") as file:
        cases = list(csv.DictReader(file))
    inputs = args.inp or PRIMARY_INPUTS
    for relative in inputs:
        matching = [row for row in cases if row["inp"].replace("\\", "/") == relative]
        if len(matching) != 1:
            raise ValueError("Expected exactly one indexed source: " + relative)
        source = (root / relative).resolve()
        source.relative_to(root)
        case = matching[0]
        source_sha = sha256(source)
        if source_sha != case["inp_sha256"]:
            raise ValueError("Indexed source hash mismatch: " + str(source))
        text = source.read_text(encoding="ascii")
        all_blocks = blocks(text)
        step_start = next(i for i, block in enumerate(all_blocks)
                          if keyword(block) == "*step")
        model = all_blocks[:step_start]
        step = all_blocks[step_start:]
        if any(keyword(block) == "*boundary" for block in model):
            raise ValueError("Initial model constraints need explicit review: " + str(source))
        if any("mass scaling" in block.splitlines()[0].lower() for block in step):
            raise ValueError("Nonuniform/explicit mass scaling needs a separate mass map: "
                             + str(source))
        boundaries = [block for block in step if keyword(block) == "*boundary"]
        boundary_rows = [line.strip() for block in boundaries for line in block.splitlines()[1:]
                         if line.strip() and not line.startswith("**")]
        expected = {"load,1,1", "load,2,2,1.", "fixity,1,1", "fixity,2,2"}
        normalized = {line.replace(" ", "").lower() for line in boundary_rows}
        if normalized != expected:
            raise ValueError("Review unfamiliar C(T) constraints: " + repr(boundary_rows))
        if any("*include" in block.lower() for block in model):
            raise ValueError("External includes need source capture")
        density = [float(numeric_line(block).split(",")[0]) for block in model
                   if keyword(block) == "*density"]
        elastic = [[float(value) for value in numeric_line(block).split(",") if value.strip()]
                   for block in model if keyword(block) == "*elastic"]
        if len(density) != 1 or len(elastic) != 1:
            raise ValueError("Expected one homogeneous material")
        # Material nonlinearity is inactive in an initial linear perturbation.
        # Remove it explicitly so this is an elastic eigenproblem in Standard.
        ignored = {"*damage initiation", "*damage evolution", "*plastic", "*amplitude"}
        prefix = "".join(block for block in model if keyword(block) not in ignored)
        controls = getattr(args, "controls", None)
        if controls is None:
            controls = ("released", "held", "unit_load") if args.stiffness else ("released", "held")
        if not controls or len(set(controls)) != len(controls) or any(control not in ("released", "held", "unit_load") for control in controls):
            raise ValueError("Select distinct released, held or unit_load controls")
        for control in controls:
            name = "{}_{}_{}_rho{}_nx{}_{}".format(
                "stiffness" if control == "unit_load" else "modal",
                case["topology"], case["l_mm"], case["nominal_relative_density"].replace(".", "p"),
                case["nnx"], control)
            directory = job_directory(out, name)
            directory.mkdir(exist_ok=True)
            inp = directory / (name + ".inp")
            suffix = ("\n** Initial undeformed elastic C(T) eigenproblem; no external load.\n"
                      "*Step, name=Modal, perturbation\n"
                      "*Frequency, eigensolver=Lanczos, normalization=mass\n"
                      "{},\n*Boundary\nfixity, 1, 2\nload, 1, 1\n").format(args.modes)
            if control == "held":
                suffix += "load, 2, 2\n"
            if control == "unit_load":
                suffix = ("\n** Initial undeformed elastic C(T) compliance; unit load in N.\n"
                          "*Step, name=Compliance, nlgeom=NO\n*Static\n1., 1.\n"
                          "*Boundary\nfixity, 1, 2\nload, 1, 1\n"
                          "*Cload\nload, 2, 1.\n")
            suffix += "*Output, field\n*Node Output\n" + (
                "U, UR, RF" if control == "unit_load" else "U, UR") + "\n*End Step\n"
            contents = prefix.rstrip() + "\n" + suffix.lstrip("\n")
            if inp.exists() and inp.read_text(encoding="ascii") != contents:
                raise ValueError("Refusing to overwrite a different modal input: " + str(inp))
            inp.write_text(contents, encoding="ascii")
            metadata = {
                "case_id": case["case_id"], "source_inp": str(source),
                "source_inp_sha256": source_sha, "input_sha256": sha256(inp),
                "source_odb": str(root / case["odb"]), "control": control,
                "job": name, "n_modes": args.modes, "analysis": (
                    "linear_static" if control == "unit_load" else "frequency"),
                "units": "N, mm, tonne, second; cycles/time is Hz",
                "density_tonne_mm3": density[0], "elastic_MPa_nu": elastic[0],
                "mass_scaling": "none in exact source INP; no modal mass adjustment",
                "boundary": "fixed pin U1/U2=0, load pin U1=0" + (
                    ", load pin U2=0 (actuator held)" if control == "held" else ", load pin U2 free"),
                "transformations": ["Retain exact original mesh, beam sections, pin couplings, E and density",
                                    "Remove amplitude, damage and plasticity; initial undeformed elastic problem",
                                    "Replace Explicit step/loads/outputs with " + (
                                        "linear static 1 N load" if control == "unit_load" else
                                        str(args.modes) + "-mode linear perturbation"),
                                    "Homogeneous pin constraints; pin rotations remain free"],
            }
            (directory / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(str(inp))


def run(args):
    for metadata in Path(args.out).glob("*/source.json"):
        data = json.loads(metadata.read_text(encoding="utf-8"))
        directory = job_directory(args.out, data["job"])
        if directory != metadata.parent.resolve():
            raise ValueError("Job name does not match its source directory")
        launcher = directory / "launcher.txt"
        if sha256(directory / (data["job"] + ".inp")) != data["input_sha256"]:
            raise ValueError("Prepared input hash changed: " + str(directory))
        if sha256(data["source_inp"]) != data["source_inp_sha256"]:
            raise ValueError("Original source hash changed: " + data["source_inp"])
        if launcher.exists():
            if "COMPLETED" not in launcher.read_text(encoding="utf-8"):
                raise RuntimeError("Existing unsuccessful attempt requires review: " + str(directory))
            print("Existing job left unchanged: " + str(directory))
            continue
        command = [args.abaqus, "job=" + data["job"], "input=" + data["job"] + ".inp",
                   "cpus=" + str(args.cpus), "memory=50%", "interactive"]
        result = subprocess.run(command, cwd=str(directory), capture_output=True, text=True)
        output = result.stdout + result.stderr
        launcher.write_text(output, encoding="utf-8")
        print(data["job"], result.returncode)
        if result.returncode or "COMPLETED" not in output:
            raise RuntimeError("Abaqus failed; inspect " + str(directory))


def extract(args):
    from odbAccess import openOdb
    import numpy as np

    summaries, static_summaries = [], []
    for metadata in Path(args.out).glob("*/source.json"):
        data = json.loads(metadata.read_text(encoding="utf-8"))
        directory = job_directory(args.out, data["job"])
        if directory != metadata.parent.resolve():
            raise ValueError("Job name does not match its source directory")
        odb_path = directory / (data["job"] + ".odb")
        launcher = directory / "launcher.txt"
        if not odb_path.exists() or not launcher.exists() or "COMPLETED" not in launcher.read_text(encoding="utf-8"):
            continue
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            if data.get("analysis") == "linear_static":
                frame = odb.steps["Compliance"].frames[-1]
                reference = odb.rootAssembly.nodeSets["LOAD"]
                values = frame.fieldOutputs["U"].getSubset(region=reference).values
                if len(values) != 1:
                    raise ValueError("Expected a single loading reference point")
                compliance = float(values[0].data[1])
                if not np.isfinite(compliance) or compliance <= 0:
                    raise ValueError("Nonpositive/nonfinite unit-load compliance")
                fixed = odb.rootAssembly.nodeSets["FIXITY"]
                reactions = frame.fieldOutputs["RF"].getSubset(region=fixed).values
                if len(reactions) != 1:
                    raise ValueError("Expected a single fixed reference point")
                reaction_y = float(reactions[0].data[1])
                if abs(reaction_y + 1.0) > 1e-5:
                    raise ValueError("Unit-load vertical reaction does not balance applied 1 N")
                data["load_N"] = 1.0
                data["fixed_pin_reaction_y_N"] = reaction_y
                data["vertical_force_balance_error_N"] = reaction_y + 1.0
                data["compliance_mm_N"] = compliance
                data["static_stiffness_N_mm"] = 1.0 / compliance
                (directory / "stiffness.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
                static_summaries.append(data)
                continue
            instance = odb.rootAssembly.instances["PART-1-1"]
            labels = np.array([node.label for node in instance.nodes], dtype=int)
            coords = np.array([node.coordinates for node in instance.nodes], dtype=float)
            order = {int(label): i for i, label in enumerate(labels)}
            elements = np.array([[order[label] for label in element.connectivity]
                                 for element in instance.elements], dtype=int)
            # Least-squares rigid translation/rotation residual tests deformability.
            xy = coords[:, :2] - coords[:, :2].mean(axis=0)
            rigid = np.zeros((2 * len(xy), 3))
            rigid[0::2, 0] = 1
            rigid[1::2, 1] = 1
            rigid[0::2, 2] = -xy[:, 1]
            rigid[1::2, 2] = xy[:, 0]
            modes, vectors = [], []
            for frame in odb.steps["Modal"].frames:
                if frame.mode <= 0 or "U" not in frame.fieldOutputs:
                    continue
                u = np.full((len(labels), 2), np.nan)
                seen = set()
                for value in frame.fieldOutputs["U"].getSubset(region=instance).values:
                    if value.nodeLabel in seen:
                        raise ValueError("Duplicate nodal displacement observation")
                    seen.add(value.nodeLabel)
                    u[order[value.nodeLabel]] = value.data[:2]
                if len(seen) != len(labels) or not np.isfinite(u).all():
                    raise ValueError("Missing/nonfinite modal nodal displacement")
                flat = u.ravel()
                fit = rigid.dot(np.linalg.lstsq(rigid, flat, rcond=None)[0])
                norm = np.linalg.norm(flat)
                residual = float(np.linalg.norm(flat - fit) / norm) if norm else 0.0
                amp2 = (u * u).sum(axis=1)
                participation = float(amp2.sum() ** 2 / (len(amp2) * (amp2 ** 2).sum()))
                modes.append({"mode": frame.mode, "frequency_Hz": frame.frequency,
                              "rigid_fit_relative_residual": residual,
                              "spatial_participation_fraction": participation,
                              "global_deformational_candidate": bool(
                                  frame.frequency > 0.1 and residual > 1e-5 and participation > 0.02)})
                vectors.append(u)
            if len(modes) != data["n_modes"]:
                raise ValueError("Requested mode count does not match complete modal output")
            candidates = [mode for mode in modes if mode["global_deformational_candidate"]]
            data["modes"] = modes
            data["candidate_mode"] = candidates[0] if candidates else None
            data["cutoff_2fn_Hz"] = 2 * candidates[0]["frequency_Hz"] if candidates else None
            data["mode_selection_limit"] = "Numerical screening; inspect mode_shapes.png before physical acceptance"
            (directory / "modes.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            np.savez_compressed(str(directory / "modes.npz"), labels=labels, coords=coords,
                                elements=elements, u=np.array(vectors),
                                frequency_Hz=np.array([mode["frequency_Hz"] for mode in modes]))
            summaries.append(data)
        finally:
            odb.close()
    (Path(args.out) / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (Path(args.out) / "static-summary.json").write_text(json.dumps(static_summaries, indent=2), encoding="utf-8")
    print("Extracted", len(summaries), "modal and", len(static_summaries), "static controls")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    for file in Path(args.out).glob("*/modes.npz"):
        data = np.load(file)
        xy = data["coords"][:, :2]
        conn = data["elements"]
        count = min(3, len(data["u"]))
        fig, axes = plt.subplots(1, count, figsize=(4.3 * count, 5), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for i, axis in enumerate(axes):
            u = data["u"][i]
            scale = np.ptp(xy, axis=0).max() * 0.07 / np.linalg.norm(u, axis=1).max()
            axis.add_collection(LineCollection(xy[conn], colors="0.85", linewidths=0.25))
            axis.add_collection(LineCollection((xy + scale * u)[conn], colors="tab:blue", linewidths=0.35))
            axis.autoscale()
            axis.set_aspect("equal")
            axis.set_title("Mode {}: {:.3f} Hz".format(i + 1, data["frequency_Hz"][i]))
            axis.set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        fig.suptitle(file.parent.name)
        fig.savefig(str(file.parent / "mode_shapes.png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "extract", "plot"))
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out", default="Z:/p1/sims/Ti/Validation/audit/modal")
    parser.add_argument("--inp", action="append", help="Case-index-relative INP; repeatable")
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--stiffness", action="store_true", help="Also prepare four 1 N elastic static controls")
    parser.add_argument("--controls", nargs="+", choices=("released", "held", "unit_load"), help="Explicit control subset; default released/held plus --stiffness")
    parser.add_argument("--abaqus", default="C:/SIMULIA/Commands/abaqus.bat")
    parser.add_argument("--cpus", type=int, default=2)
    args = parser.parse_args()
    if args.modes < 1 or args.cpus < 1:
        parser.error("--modes and --cpus must be positive integers")
    globals()[args.action](args)


if __name__ == "__main__":
    main()

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

POPULATION_SIZES = {"tri": 30, "kagome": 20, "hex": 20, "FCC": 20}


def periodic_contents(source, task, modes=10):
    """Retain exact periodic source model; replace only nonlinear step/material.

    UAT grips are held in their prescribed incremental displacement degrees of
    freedom. C(T) retains the established released loading-pin natural-mode
    convention. These are different specimen constraints, recorded explicitly.
    """
    if task not in ("UT", "FT") or modes < 1:
        raise ValueError("Expected UT/FT and positive mode count")
    all_blocks = blocks(Path(source).read_text(encoding="ascii"))
    split = next(i for i, b in enumerate(all_blocks) if keyword(b) == "*step")
    model, step = all_blocks[:split], all_blocks[split:]
    if any(keyword(b) in ("*boundary", "*include") for b in model):
        raise ValueError("Initial constraints/includes require source review")
    if any("mass scaling" in b.splitlines()[0].lower() for b in all_blocks):
        raise ValueError("Mass-scaled sources are not an exact physical eigenproblem")
    observed = {line.replace(" ", "").lower() for b in step if keyword(b) == "*boundary"
                for line in b.splitlines()[1:] if line.strip() and not line.startswith("**")}
    expected = ({"load,1,1", "load,2,2,1.", "fixity,1,1", "fixity,2,2"} if task == "FT" else
                {"set-top,1,1", "set-top,2,2,1.", "set-top,6,6", "set-bottom,1,1",
                 "set-bottom,2,2", "set-bottom,6,6"})
    if observed != expected:
        raise ValueError("Unfamiliar exact-source constraints: " + repr(observed))
    density = [float(numeric_line(b).split(",")[0]) for b in model if keyword(b) == "*density"]
    elastic = [[float(v) for v in numeric_line(b).split(",") if v.strip()]
               for b in model if keyword(b) == "*elastic"]
    if len(density) != 1 or len(elastic) != 1 or density[0] <= 0:
        raise ValueError("One physical density and elastic material required")
    removed = {"*plastic", "*damage initiation", "*damage evolution", "*amplitude"}
    prefix = "".join(b for b in model if keyword(b) not in removed)
    constraints = ("fixity, 1, 2\nload, 1, 1\n" if task == "FT" else
                   "Set-bottom, 1, 2\nSet-bottom, 6, 6\nSet-top, 1, 2\nSet-top, 6, 6\n")
    suffix = ("\n*Step, name=Modal, perturbation\n"
              "*Frequency, eigensolver=Lanczos, normalization=mass\n{},\n"
              "*Boundary\n{}*Output, field\n*Node Output\nU, UR\n*End Step\n").format(modes, constraints)
    return prefix.rstrip() + "\n" + suffix.lstrip("\n"), {
        "density_tonne_mm3": density[0], "elastic_MPa_nu": elastic[0],
        "control": "released" if task == "FT" else "held_grips",
        "boundary": ("fixed pin U1/U2; loading pin U1 held, U2 released; pin rotations free" if task == "FT" else
                     "both source clamps U1/U2/UR3 held; homogeneous incremental prescribed travel"),
        "model_sha256": hashlib.sha256(prefix.encode("ascii")).hexdigest(),
        "transformations": ["Exact source mesh, all beam sections including UAT brackets, material E/nu/density and couplings retained",
                            "Plasticity, damage and amplitude removed for undeformed elastic perturbation",
                            "Explicit step replaced by 10-mode Lanczos frequency extraction"],
    }


def prepare_periodic(args):
    """Prepare exactly eight population periodic families; never submit."""
    archive, out = Path(args.archive).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records, jobs = [], []
    for topology, nx in POPULATION_SIZES.items():
        for task, label in (("UT", "Ductile"), ("FT", "Fracture")):
            matches = list((archive / topology).glob("*/zip/{}-{}-{}-per-1.inp".format(label, topology, nx)))
            if len(matches) != 1:
                raise ValueError("Expected one population periodic source for {} {}: {}".format(topology, task, matches))
            source = matches[0].resolve()
            source.relative_to(archive)
            name = "pm_{}_{}".format(topology.lower(), task.lower())
            directory = job_directory(out, name)
            directory.mkdir(exist_ok=True)
            inp = directory / (name + ".inp")
            contents, data = periodic_contents(source, task, args.modes)
            if inp.exists() and inp.read_text(encoding="ascii") != contents:
                raise ValueError("Prepared periodic modal input is immutable: " + str(inp))
            inp.write_text(contents, encoding="ascii")
            data.update(job=name, case_id="population-periodic:{}:{}".format(topology, task),
                        task=task, topology=topology, nnx=nx, l_mm=10., nominal_relative_density=.2,
                        source_inp=str(source), source_inp_sha256=sha256(source),
                        source_odb=str(source.with_suffix(".odb")), input_sha256=sha256(inp),
                        n_modes=args.modes, analysis="frequency", mass_scaling="none",
                        units="N, mm, tonne, second; frequency in Hz",
                        disorder_application="same periodic cutoff for every matching disorder ID; no disordered modal jobs")
            (directory / "source.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            records.append(data)
            jobs.append({"job":name, "input":str(inp), "input_sha256":data["input_sha256"],
                         "cpus":8, "required_outputs":[".odb", ".dat"]})
    manifest = {"abaqus":args.abaqus, "queue_root":str(out / "queue"), "jobs":jobs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "prepared-periodic.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(str(out / "manifest.json"))


def prepare_iso(args):
    """Prepare missing exact periodic modal sources for selected native FT points."""
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (root/"audit/paper/points.csv").open(newline="") as stream:
        selected = {r["source_inp_sha256"]: r for r in csv.DictReader(stream)
                    if r.get("method") == "Hsieh_initial_triangle_native" and r.get("plotted") == "True"}
    with (root/"case-index.csv").open(newline="",encoding="utf-8-sig") as stream:
        indexed = {r["inp_sha256"]:r for r in csv.DictReader(stream)}
    existing = json.loads((root/"audit/modal/summary.json").read_text())
    reused = {r["source_inp_sha256"]:r for r in existing if r["control"] == "released"}
    records, jobs = [], []
    for source_sha, row in selected.items():
        if source_sha in reused:
            continue
        case = indexed.get(source_sha,{})
        source = Path(row["source_inp"]) if row.get("source_inp") else root/case["inp"]
        source_odb = Path(row["source_odb"]) if row.get("source_odb") else root/case["odb"]
        if sha256(source) != source_sha:
            raise ValueError("Selected original input hash mismatch")
        length = float(row["l_mm"])
        rho = float(row["relative_density"])
        topology = row["topology"]
        name = "im_{}{}_r{}".format({"tri":"t","kagome":"k","hex":"h"}[topology],int(length),round(rho*100))
        directory = job_directory(out,name)
        directory.mkdir(exist_ok=True)
        inp = directory/(name+".inp")
        contents, metadata = periodic_contents(source,"FT",args.modes)
        if inp.exists() and inp.read_text(encoding="ascii") != contents:
            raise ValueError("Prepared ISO modal input is immutable")
        inp.write_text(contents,encoding="ascii")
        metadata.update(job=name,case_id=row["case_id"],task="FT",topology=topology,l_mm=length,
            nominal_relative_density=rho,source_inp=str(source),source_inp_sha256=source_sha,
            source_odb=str(source_odb),source_odb_sha256=row.get("source_odb_sha256") or case["odb_sha256"],
            input_sha256=sha256(inp),n_modes=args.modes,analysis="frequency",mass_scaling="none",
            units="N, mm, tonne, second; frequency in Hz",
            scope="Exact selected periodic validation specimen; no disorder-specific modal solves")
        (directory/"source.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        records.append(metadata)
        jobs.append(dict(job=name,input=str(inp),input_sha256=metadata["input_sha256"],cpus=8,required_outputs=[".odb",".dat"]))
    (out/"manifest.json").write_text(json.dumps(dict(abaqus=args.abaqus,queue_root=str(out/"queue"),jobs=jobs),indent=2))
    (out/"prepared-iso.json").write_text(json.dumps(records,indent=2))
    print("Prepared",len(jobs),"periodic jobs; retained",len(reused),"existing exact sources")
    print(str(out/"manifest.json"))


def iso_map(args):
    """One original-INP-hash map for completed source-qualified released modes."""
    root,out=Path(args.root),Path(args.out)
    old=json.loads((root/"audit/modal/summary.json").read_text())
    new=json.loads((out/"summary.json").read_text()) if (out/"summary.json").is_file() else []
    mapping={}
    for data in old+new:
        if data["control"] != "released":
            continue
        folder=(out/data["job"] if data in new else root/"audit/modal"/data["job"])
        if not data.get("candidate_mode") or not (folder/"mode_shapes.png").is_file():
            raise ValueError("Modal physical screen/shape review absent")
        source=Path(data["source_inp"])
        if sha256(source) != data["source_inp_sha256"]:
            raise ValueError("Original INP changed")
        original_odb_sha=sha256(data["source_odb"])
        if data.get("source_odb_sha256") and original_odb_sha != data["source_odb_sha256"]:
            raise ValueError("Original fracture ODB changed")
        record=folder/"modes.json"
        modal_odb=Path(data.get("solved_odb") or folder/(data["job"]+".odb"))
        modal_odb_sha=sha256(modal_odb)
        if data.get("solved_odb_sha256") and modal_odb_sha != data["solved_odb_sha256"]:
            raise ValueError("Extracted modal ODB changed")
        mapping[data["source_inp_sha256"]]=dict(natural_frequency_Hz=data["candidate_mode"]["frequency_Hz"],
            cutoff_Hz=data["cutoff_2fn_Hz"],job=data["job"],source_sha256=data["source_inp_sha256"],
            source_odb_sha256=original_odb_sha,read_only=True,state="completed_source_qualified_released_physical_mode",
            source_input_hash_verified=True,source_inp=str(source),source_odb=data["source_odb"],
            modal_receipt=str(record),modal_receipt_sha256=sha256(record),
            modal_odb=str(modal_odb),modal_odb_sha256=modal_odb_sha,boundary=data["boundary"],
            execution_state=data.get("execution_state","completed"),
            delivery_receipt=data.get("delivery_receipt"),
            delivery_receipt_sha256=data.get("delivery_receipt_sha256"),
            physical_mode_review="Lowest global deformational mode; mode shapes visually reviewed",mode=data["candidate_mode"]["mode"])
    out.mkdir(parents=True,exist_ok=True)
    path=out/"source-frequency-map.json"
    path.write_text(json.dumps(mapping,indent=2)+"\n")
    print("Mapped",len(mapping),"exact periodic original sources:",str(path))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                case["topology"], case["l_mm"].replace(".", "p"), case["nominal_relative_density"].replace(".", "p"),
                case["nnx"], control)
            # SiC batch identity is unique within this indexed validation set;
            # compact names avoid Abaqus' Windows restart-file path limit.
            if (case.get("campaign") == "Orthotropic SiC validation"
                    and case["topology"] in ("square", "45square") and case.get("batch") in ("A", "B", "C", "D")):
                name = ("stiff_{}_{}".format(case["topology"], case["batch"]) if control == "unit_load"
                        else "modal_{}_{}_{}".format(case["topology"], case["batch"], control))
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
        delivery_root = getattr(args, "deliveries", None)
        delivery_roots=([delivery_root]+(getattr(args,"extra_deliveries",None) or [])) if delivery_root else []
        matches=[Path(item)/data["job"] for item in delivery_roots
                 if (Path(item)/data["job"]/(data["job"]+".odb")).is_file()]
        if len(matches)>1:
            raise ValueError("Multiple delivered modal attempts require an explicit source choice")
        solved=(matches[0] if matches else Path(delivery_root)/data["job"]) if delivery_root else directory
        odb_path = solved / (data["job"] + ".odb")
        launcher = solved / "launcher.txt"
        if not odb_path.exists() or not launcher.exists():
            continue
        launch_text=launcher.read_text(encoding="utf-8")
        completed="COMPLETED" in launch_text.upper()
        reviewed=data["job"] in (getattr(args,"accept_wrapup_job",None) or [])
        if not completed and not reviewed:
            continue
        if reviewed and not completed:
            dat=(solved/(data["job"]+".dat")).read_text(encoding="utf-8",errors="replace")
            if ("THE ANALYSIS HAS BEEN COMPLETED" not in dat or "ANALYSIS COMPLETE" not in dat
                    or "End Abaqus/Standard Analysis" not in launch_text
                    or "SIM Wrap-up" not in launch_text or "MoveFileEx: Access is denied" not in launch_text):
                raise ValueError("Only reviewed post-analysis SIM wrap failure may be extracted")
        data["execution_state"]=("completed" if completed else "mechanically_completed_with_reviewed_SIM_wrap_failure")
        if delivery_root:
            delivery=solved/"delivery.json"
            receipt=json.loads(delivery.read_text())
            if receipt.get("required_outputs_missing"):
                raise ValueError("Delivery is incomplete")
            for suffix in (".inp",".odb",".dat"):
                expected=next(r for r in receipt["files"] if r["file"]==data["job"]+suffix)
                if sha256(solved/expected["file"]) != expected["sha256"]:
                    raise ValueError("Retained delivery hash changed")
            data["delivery_receipt"]=str(delivery)
            data["delivery_receipt_sha256"]=sha256(delivery)
        if sha256(data["source_inp"]) != data["source_inp_sha256"]:
            raise ValueError("Source input changed after preparation")
        if sha256(solved / (data["job"] + ".inp")) != data["input_sha256"]:
            raise ValueError("Solved input differs from prepared input")
        before_odb = sha256(odb_path)
        data.update(solved_odb=str(odb_path), solved_odb_sha256=before_odb)
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
        if sha256(odb_path) != before_odb:
            raise ValueError("Read-only modal extraction changed ODB")
    (Path(args.out) / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (Path(args.out) / "static-summary.json").write_text(json.dumps(static_summaries, indent=2), encoding="utf-8")
    print("Extracted", len(summaries), "modal and", len(static_summaries), "static controls")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    files=list(Path(args.out).glob("*/modes.npz"))
    for file in files:
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
    if files:
        rows=(len(files)+2)//3
        fig,axes=plt.subplots(rows,3,figsize=(11.,3.6*rows),squeeze=False,constrained_layout=True)
        for axis in axes.ravel():
            axis.set_visible(False)
        for axis,file in zip(axes.ravel(),files):
            data=np.load(file)
            xy,conn=data["coords"][:,:2],data["elements"]
            u=data["u"][0]
            scale=np.ptp(xy,axis=0).max()*.07/np.linalg.norm(u,axis=1).max()
            axis.set_visible(True)
            axis.add_collection(LineCollection(xy[conn],colors=".86",linewidths=.18))
            axis.add_collection(LineCollection((xy+scale*u)[conn],colors="tab:blue",linewidths=.25))
            axis.autoscale()
            axis.set_aspect("equal")
            axis.set_title(file.parent.name+"\nMode1: {:.3f} Hz".format(data["frequency_Hz"][0]),fontsize=9)
            axis.tick_params(labelsize=7)
        fig.savefig(str(Path(args.out)/"mode_overview.png"),dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare_periodic", "prepare_iso", "iso_map", "run", "extract", "plot"))
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out", default="Z:/p1/sims/Ti/Validation/audit/modal")
    parser.add_argument("--archive", default="Z:/p1/HPCarchive/Ti/data/per/0.0")
    parser.add_argument("--deliveries", help="Serial queue runs directory for hash-bound read-only extraction")
    parser.add_argument("--accept-wrapup-job",action="append",help="After explicit review only: extract mechanically completed job with known SIM MoveFileEx wrap failure; queue failure remains recorded")
    parser.add_argument("--extra-deliveries",action="append",help="Additional distinct continuation queue runs directory; repeated job attempts are rejected")
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

# p1 Abaqus And HPC Agent Context

This directory contains the active Abaqus model-generation, post-processing, and transfer scripts for p1. It is the source of the raw FEA artifacts consumed by p1 notebooks and p2 ML workflows.

## Active Script Roles

- `A1_FractureToughness-Ductility.py` is the main Abaqus CAE model builder. It creates ductile/uniaxial and fracture/compact-tension models, applies nodal or strut disorder, writes or submits jobs, and appends frequency metadata when needed.
- `A2_INpostProcess.py` parses generated `.inp` files and exports input-side transfer files for nodes, struts, and frequency-disorder parameters.
- `A2_OUTpostProcess.py` parses Abaqus `.odb` history outputs and exports curve-side `OUT-Ductile...csv` and `OUT-Fracture...csv` files.
- `A2_FieldOUTpostProcess.py` parses Abaqus `.odb` field outputs and exports per-simulation `FIELDu-...npz` files using matching `.inp` node labels.
- `A3_ContinuumPP.py` is an older continuum/displacement post-processing helper that exports `frame*.csv` and `NodesElems.csv` style outputs.
- `B1_ABAQUS-new.sh` is the main Slurm submit wrapper. It stages code/resources to scratch, runs A1 and A2 scripts, then archives logs, inputs, outputs, resources, and transfer files.
- `B1_ABAQUS-inp-rerun.sh` runs existing `.inp` files from the current directory and can post-process them in place.
- `B2_ABAQUS-PPscratch-new.sh` re-runs post-processing over archived Abaqus job directories. It can be submitted from one lattice archive directory or from a parent disorder directory such as `.../disNodes/0.2`; it discovers `zip/` directories recursively and skips only directories without `.odb` files.
- `B3_ABAQUS-transfer.sh` moves `transfer/` files and archives between QMUL HPC and local `Z:/p1/data/Ti`.
- `run-local.ps1` is a local Abaqus launcher for Windows testing.
- `odbUpgrade.py` upgrades ODB files and can delete old ODBs when configured. Treat it as potentially destructive.

## Historical Folders

- `OldScriptVersions/` and `OldScriptVersions/HPC-UGEscripts/` are historical references, including older Abaqus, UGE, and Akash-origin scripts.
- Do not edit or run historical scripts unless the user explicitly asks for archaeology or migration work.
- Use historical folders to preserve old experiments. Do not keep stale branches, abandoned approaches, or superseded code inside active A/B scripts.
- Ignore `__pycache__/` files.

## Abaqus Command Contract

The active A1/A2 scripts share a positional argument contract after `--`:

```text
LAT nnx unitCellSize mode material rD DIS fac distribution target initial nJobs CPUs Fout Hout pDir
```

Keep this order stable unless you also update every launcher and downstream script. Some workflows also use optional optimization or stiffness-matrix arguments.

Important meanings:

- `LAT`: lattice type, commonly `FCC`, `tri`, `kagome`, or `hex`.
- `mode`: `ductile`, `fracture`, `both`, or script-specific `any`.
- `material`: currently includes material law branches such as `Ti`, `Al`, and `SiC`.
- `DIS`: `per`, `disNodes`, or `disStruts`.
- `fac`: disorder magnitude, with processed names often encoding `fac * 100`.
- `distribution`: examples include `uniform`, `lhs`, `lhs_uniform`, `frequency`, `opt-f`, `normal`, and `exponential`.
- `target`: disorder-node targeting such as `all` or axis-specific variants.
- `pDir`: working directory/archive directory containing generated Abaqus files and `transfer/`.

## Naming And Transfer Files

- Ductile/uniaxial files use the `Ductile` token; fracture/compact-tension files use `Fracture`.
- Periodic/reference runs use `per` and are treated downstream as sample id `0`.
- Node-disorder input exports use `IN-n...csv`; strut-disorder exports use `IN-s...csv`; frequency parameters use `IN-f...csv`.
- Curve outputs use `OUT-Ductile...csv` or `OUT-Fracture...csv`.
- Field outputs use `FIELDu-{job_stem}.npz` and must have a matching `.inp` so node labels and coordinates can be recovered.
- `A2_FieldOUTpostProcess.py` should preserve the node labels read from the matching `.inp` node block. Do not apply the old fracture-only deletion of two nodes in the ML field pipeline unless the `.inp` node block itself again contains reference points.
- The p1 notebooks expect these names; do not change them casually.
- Current local raw transfer storage is separated by case: periodic/reference files go to `Z:/p1/data/Ti/per/{LAT}/transfer`, while disordered files go to paths such as `Z:/p1/data/Ti/disNodes/{fac}/{LAT}/transfer` or `Z:/p1/data/Ti/disNodes/{PATH_EXTRA}/{fac}/{LAT}/transfer`.

## Abaqus Model Details

- `A1_FractureToughness-Ductility.py` depends on `resources.lattices.Geometry`, `connectivity`, `insidePoint`, `pStrainProperties`, and `resources.abaqus.node`.
- Ductile models create bracket/body node sets, beam sections, ExplicitDynamics steps, top/bottom/loading regions, history outputs, and field outputs.
- Fracture models create compact-tension/crack geometry, crack-tip output regions, reference-point couplings, displacement loading, and status outputs.
- Stiffness-matrix branches use special boundary conditions and output paths; do not simplify them into the standard ductile path.
- Frequency-disorder runs append `**FREQUENCIES` metadata to the `.inp`; `A2_INpostProcess.py` relies on that block.

## HPC Rules

- `B1_ABAQUS-new.sh` stages scripts and `resources/` into scratch and archives to `/data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data`.
- Preserve scratch path safety checks before deleting scratch content.
- Preserve run configuration capture (`run_config`) and the staged `resources/` copy; archived runs need provenance.
- `PATH_EXTRA` changes archive layout for targeted data such as `Target-xs`; keep local and remote transfer path rules aligned with `B3_ABAQUS-transfer.sh`.
- Do not change Slurm partition, account, license, CPU, memory, or time directives unless the user gives the target policy.

## How To Work Here

- Use Abaqus Python for real execution of scripts that touch `mdb`, `session`, or `openOdb`.
- Run `python tools/validate_repo.py --changed` from the repository root. Its Python result is syntax-only for Abaqus-dependent files and does not prove Abaqus API behavior.
- If changing A1/A2 names or output contracts, use `review-p1-p2-data-contract` to inspect every affected producer and consumer.
- If changing ODB parsing, verify expected step counts (`Hout + 1` in current launchers) and whether missing history data should be zero-filled or extrapolated.
- If changing field output, use the field schema and impact map in the data-contract skill reference.
- Keep active scripts direct and intentional. Avoid adding broad fallback layers or compatibility scaffolding unless a current launcher or archived data path concretely needs it.
- Treat `odbUpgrade.py`, backup deletion, renaming utilities, and recursive archive scripts as destructive unless run in dry-run mode or on a confirmed target.

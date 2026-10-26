# p1 Abaqus And HPC Agent Context

This directory contains the active Abaqus model-generation, post-processing, and transfer scripts for p1. It is the source of the raw FEA artifacts consumed by p1 notebooks and p2 ML workflows.

## Active Script Roles

- `A1_FractureToughness-Ductility.py` is the main Abaqus CAE model builder. It creates ductile/uniaxial and fracture/compact-tension models, applies nodal or strut disorder, writes or submits jobs, and appends frequency metadata when needed.
- `A2_INpostProcess.py` parses generated `.inp` files and exports input-side transfer files for nodes, struts, and frequency-disorder parameters.
- `A2_OUTpostProcess.py` parses Abaqus `.odb` history outputs and exports curve-side `OUT-Ductile...csv` and `OUT-Fracture...csv` files.
- `A2_FieldOUTpostProcess.py` parses Abaqus `.odb` field outputs and exports per-simulation `FIELDu-...npz` files using matching `.inp` node labels.
- `A2_DamageOUTpostProcess.py` is the opt-in single-file strut exporter. Its named arguments are separate from the legacy positional A1/A2 contract. The full audit requires an explicit partial rule; `--status-only` writes lean `strut_status_v1` products without SDEG/U/history dependencies. Use Abaqus Python and matching supported periodic or 0.2 disNodes/lhs/all Ti FCC20 B21 ODB/INP paths. It never upgrades ODBs or submits jobs. `resources/damage.py` validates original chains; whole-element STATUS=0 establishes failure. UT selects BodySet, excluding grips. Missing observations remain unknown. Preserve initial geometry, simulation IDs, source/code hashes and original archive paths when reading scratch copies. Disordered mapping requires degree-based original joints, unique bounded reference identities and exact canonical edges.
- `A2_DamageBatch.py` prepares an explicit FCC-only inventory and executes reviewed batches through the exporter. The authorised production roots are Ti `disNodes/0.2/FCC` and its separate periodic FCC baseline. Pin the manifest/code fingerprints; stage bounded ODB/INP copies in Slurm-owned scratch; validate corresponding FIELDu metadata by sample identity, geometry and frame order/count; publish only validated `FIELDd` NPZ/JSON pairs. Small per-frame time differences are diagnostic, not an equality gate. Resume only verified matching outputs; retain per-file failure evidence. This is post-processing of existing results, never new FEA.
- `A3_ContinuumPP.py` is an older continuum/displacement post-processing helper that exports `frame*.csv` and `NodesElems.csv` style outputs.
- `B1_ABAQUS-new.sh` is the main Slurm submit wrapper. It stages code/resources to scratch, runs A1 and A2 scripts, then archives logs, inputs, outputs, resources, and transfer files.
- `B1_ABAQUS-inp-rerun.sh` runs existing `.inp` files from the current directory and can post-process them in place.
- `B2_ABAQUS-PPscratch-new.sh` retains its legacy all-export route. Its explicit `--damage-only --manifest ...` route runs only the reviewed FCC batch in scratch, with a genuinely read-only dry run. Do not use the recursive legacy route for damage processing or submit from a parent containing other lattices.
- `B3_ABAQUS-transfer.sh` moves `transfer/` files and archives between QMUL HPC and local `Z:/p1/data/Ti`.
- `B3_ABAQUS-archive.sh` is the separate Windows/Git Bash full-p1 download: `/data/SEMS-TaoLab/Niccolo-Forte/p1/` contents to external `Z:/p1/HPCarchive/`, without flattening or an extra p1 parent. It uses installed Python/PuTTY and the authenticated shared HPC session, never uploads/deletes/extracts/submits. `--dry-run` is entirely offline; `--check-only` compares metadata. Use `get` for missing/empty files and `reget` only for nonempty partial files; PSFTP cannot reget a nonexistent local file. Local batch paths use forward slashes. Per-file resume requires an unchanged source and its sibling `.resume.json`, removed on success. Foreground execution reports actual local byte progress every five seconds; Ctrl+C stops only the transfer client, retaining partials/state and the shared session. Never run simultaneous copies. Source type/name/case checks reject Windows-incompatible entries rather than silently omitting them. Paths/types/sizes are checked, not content hashes or Unix permissions; do not treat success as authority to delete HPC originals. Validate with `test_archive_transfer.py` mocks; any live validation must be explicitly bounded, not a full download unless requested.
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
- Production binary-fracture outputs use lowercase `FIELDd-{job_stem}.npz` beside FIELDu in each original HPC `<job>/zip/transfer/`. Local transfer consolidates into exactly `Z:/p1/data/Ti/per/FCC/transfer` or `Z:/p1/data/Ti/disNodes/0.2/FCC/transfer`; retain source job/path provenance rather than mirroring job directories. Keep curated audit ODB/INP examples in their existing `sims/` location. The four additive MLdata banks live locally under disNodes/0.2/FCC/MLdata and remotely under `/data/SEMS-TaoLab/Niccolo-Forte/p2/MLdata`.
- Generated p2 damage reports/examples belong only in `Z:/00-PhD-gitRepo/p2-DisorderML/samples/`, not other checkouts. This does not relocate upstream p1 data.
- Damage outputs use the separate `strut_damage_v1` NPZ contract, not FIELDu/allFIELD or node targets. Retain the initial frame and actual step/frame/time, native task struts, original instance/element membership, explicit audit-canonical endpoint order and masks. In the checked ODBs, only section points 1/5 are reported; their maximum is a severity summary, not average cross-section degradation. A point SDEG=1 does not itself prove whole-element failure; use the validated STATUS convention. No near-one threshold defines failure.
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

- The user-curated validation subset now lives at `Z:/p1/sims/Ti/Validation`,
  not `HPCarchive/Ti/validation`. Its file manifest maps every original local/HPC
  path to retained content or an explicit cleanup decision. The old archive
  directory contains a relocation notice. Historical full-mirror counts refer
  to the pre-curation snapshot; do not restore discarded validation scratch or
  CPU-repeat ODBs merely to make those counts match. Remote originals were not
  modified by this local consolidation. Captured scripts under validation
  `source/` are historical execution evidence, not maintained entry points.
- The preserved full archive is `Z:/p1/HPCarchive`: map an original HPC p1-relative
  path directly beneath it, e.g. `p1/Ti/data/...` -> `HPCarchive/Ti/data/...`.
  This is not the consolidated `Z:/p1/data/Ti` processing tree. Use explicit local
  paths when revisiting archived ODB/INP/execution files. Keep original paths in
  saved provenance; do not rewrite immutable manifests or replace HPC paths with
  Windows strings. If the researcher later removes the remote p1 archive, an HPC
  reprocessing pass must first restore its required subset. Existing p1 producer
  output defaults remain valid for new runs; p2 training uses its separate p2/MLdata.
- `B1_ABAQUS-new.sh` stages scripts and `resources/` into scratch and archives to `/data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data`.
- Preserve scratch path safety checks before deleting scratch content.
- Preserve run configuration capture (`run_config`) and the staged `resources/` copy; archived runs need provenance.
- After jobs complete, home retains successful output logs only; copies may accompany archived results. Keep code, manifests and necessary audit records in the existing run archive, without a provenance tier. MLdata and its subfolders contain only files directly consumed by ML training/testing.
- `PATH_EXTRA` changes archive layout for targeted data such as `Target-xs`; keep local and remote transfer path rules aligned with `B3_ABAQUS-transfer.sh`.
- Do not change Slurm partition, account, license, CPU, memory, or time directives unless the user gives the target policy.

## How To Work Here

- Windows examples in `DAMAGE.md` run from the primary `Z:/00-PhD-gitRepo` checkout with the selected standard Python environment; ODB extraction still requires Abaqus Python.
- Use Abaqus Python for real execution of scripts that touch `mdb`, `session`, or `openOdb`.
- Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed` from the repository root. Its Python result is syntax-only for Abaqus-dependent files and does not prove Abaqus API behavior.
- Damage checks: run the focused damage/status tests through unittest discovery; run `test_damage_odb.py --npz PATH` in Abaqus Python for independent every-frame raw STATUS and parent failure/known-mask comparisons. It accepts full-audit and STATUS-only schemas; only full audits require SDEG spot checks. Raw FIELDu validation matches INP instance names case-insensitively, as Abaqus does, and rejects ambiguous case collisions. `DAMAGE.md` owns commands and schemas. Syntax validation is not a substitute for live read-only checks.
- If changing A1/A2 names or output contracts, use `review-p1-p2-data-contract` to inspect every affected producer and consumer.
- If changing ODB parsing, verify expected step counts (`Hout + 1` in current launchers) and whether missing history data should be zero-filled or extrapolated.
- If changing field output, use the field schema and impact map in the data-contract skill reference.
- Keep active scripts direct and intentional. Avoid adding broad fallback layers or compatibility scaffolding unless a current launcher or archived data path concretely needs it.
- Treat `odbUpgrade.py`, backup deletion, renaming utilities, and recursive archive scripts as destructive unless run in dry-run mode or on a confirmed target.

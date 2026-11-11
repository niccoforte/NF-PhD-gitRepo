# p1 Notebook Agent Context

This directory contains the notebook layer for p1: FEA-derived mechanics, data processing, validation, and exploratory analysis for quasi-disordered 2D lattices. Most notebooks depend on shared helpers in `resources/`; read the helper code before changing notebook behavior.

## Workflow Role

- `p1-DisorderLatticeProperties/SIMscripts/` generates Abaqus models and writes raw `transfer/` files.
- `DataProcessing.ipynb` is the current batch processing notebook. It builds `DATA(path=1, ...)` objects, loops over lattice/disorder settings, and calls `create_inputCSV`, `create_outputCSV`, and `create_fieldNPZ`.
- `InputsOutputs.ipynb` inspects and saves ML-ready curve and field datasets using `load_data`, `UTprops`, `FTprops`, `MULTIprops`, `save_MLdata`, `save_MULTIdata`, `save_field_MLdata`, and `save_MULTIfieldData`.
- `SIMresults.ipynb` is for quick local/transfer result inspection and mechanical-property calculation from output CSVs.
- `StiffnessMatrix.ipynb` handles analytical/simulation stiffness matrices, isotropy checks, and effective-property calculations.
- `ContinuumPlots.ipynb` and `ValConvPlots.ipynb` support displacement-field visualization, material/validation plots, mesh/refinement checks, and convergence/validation figures.
- Continuum principal plots show both eigenvalues of reconstructed Green-Lagrange
  strain with x/y coordinates and a shared signed scale. Confirm saved sample/frame
  provenance before exporting an initiation figure; the colour cap is not a
  beam-section damage threshold. Keep historical manuscript assets until matched.
- Native history/failure audit helpers preserve original ODBs, timestamps,
  validity, STATUS and named energies. `validation_benchmark.py` digitises Hsieh
  Fig10; only3mm is direct,10mm scaling is declared. `validation_comparison.py`
  joins saved identities. Workbook mirroring is additive and refuses occupied
  rows; preserve its verified backup and canonical reference CSV.
  `validation_queue.py` alone submits reviewed immutable manifests,8CPUs serial,
  verifies delivery and cleans only owned scratch; each continuation has a
  distinct queue root. Failed packaging remains failed pending pinned review.
  Modal actions prepare_periodic/prepare_iso cover population8 and selected
  ISO sources; iso_map joins verified frequencies. Use causal
  order2/DC Butterworth at2fn for matching disorder, no disordered modal jobs.
  A cutoff at/above saved Nyquist is unresolved, never silently capped; preserve
  native values with truthful filter provenance. MLdata is not rewritten here.
  `validation_fcc.py` owns source-mesh3crack elastic controls, one mapped periodic
  tensor and6nonlinear feasibility jobs. Potential energy derivatives remain
  conditional on constitutive/local/global agreement; no automatic plastic-J
  qualification or population reruns. Use full pin opening, no half-model factor.
  `validation_literature.py` compares Omidi2023 snub-square TableC2 geometry
  factors with Ortega; this is not completed discrete-beam fracture validation.
- `validation_failure_audit.py` binds each of the48 native tensile-strain
  initiation brackets to saved section damage and source fingerprints.
  `validation_controls.py` prepares/extracts exact-source B21 conversion and
  initial-compliance controls.
  `prepare_uat_linear` retains exact UAT clamps/sections and computes the initial
  tangent by elastic travel/reaction, never a fit through unresolved plasticity.
  A body-gauge slope is not the full conversion tensor. No D Mohr fallback.
  `validation_work_partition.py` measures matched nonlinear unloading separately
  from native IE-SE dissipation estimates.
  `validation_results.py` verifies extraction attestations, event indices/times,
  undamaged prefixes and source-specific controls before writing the selected
  comparison CSV. Hsieh Appendix D uses an initial-slope triangle: negative
  area remains unavailable. Nonlinear recovery and dissipation eta-J estimates
  remain distinct methods, never replacements falsely attributed to Hsieh.
  `validation_inventory.py` inventories retained evidence and named task scratch;
  its `cleanup-review.json` also covers workbook additions/legacy values, code,
  early reviews, pending reference capture and temporary evidence dependencies.
  It never deletes. Continuation `-status.json` receipts are not submission plans.
  Refresh it before final cleanup.
- Final figures require explicit user approval before any change. Default
  comparison refreshes never write them; --write-paper is approval-only.
  The approved pre-filter display is restored:14native,2recovery,2predictions.
  Its hash-bound recipe is audit/paper/approved-figure-recipe.json; forward
  Butterworth calculations remain separate. Preserve every point's provenance.
  `audit/paper` holds PNG/PDF/SVG, point/exclusion tables and captions. Fig10
  uses nominal density, direct3mm references and declared10mm length scaling.
  Fig8b sqrt3 views alter display ordinates only, with factors/captions; base K
  remains unchanged and neither hypothesis is a verified correction.
  Use Hsieh et al./Current study source legends, hollow references, layered
  Current study circles and direct topology labels without coordinate jitter.
  Internal Simulation IDs and point provenance remain intact.
- `validation_completion.py` prepares the seven absent length/density cases
  under `audit/new-validation`, using archived A1 INPs and verified dimensional
  scaling rather than today's A1 defaults. It never submits; the serial queue
  owns execution. Its prepared catalogue is not evidence of completed results.
  Preserve the immutable original index; qualify new source hashes, initiation,
  KE/IE, modulus and compliance before including completed cases in plots.
  Hexagonal13% nx80 is anchored to the largest retained hex mesh, not new PSC
  convergence evidence. History output uses supported time intervals.
  Submitted preparation is immutable for both default and refresh calls;
  preflight uses the durable serial-owner gate as well as the OS lock, so an
  interrupted worker cannot allow a second solver to start unnoticed.
  `--extract-finished` creates an additive results catalogue using native
  history/strain/damage exporters. `--include-interrupted` requires a dead-owner
  review before reading a partial ODB. Source fingerprints bracket the complete
  extraction batch; event/table/export hashes pin the settled result. Failed
  post-initiation tails do not invalidate an independently verified intact prefix.
  Endpoint KE/IE is not full-path quasistatic qualification. New-case tables and
  captions expose transient-review flags; slower-rate checks remain separate.
  Results merge only qualified cases; restarts use frozen parent generation
  records while retaining actual attempt/ODB identities. Uniform scaling permits
  linear-control reuse (C unchanged, k scales with length) after source/material/
  mesh/BC/periodic-patch proof; never use it for nonlinear fracture or loading
  rate equivalence. New13% hexagonal controls remain separate supplemental solves.
- `AK-DataProcessing.ipynb` and `AK-InputsOutputs.ipynb` are legacy Akash-data workflows using `DATA(path=0, ...)`; do not mix them into the current Ti data pipeline unless the user asks.
- `FunctionApproximation.ipynb`, `Sampling.ipynb`, and `QuickPlots..ipynb` are exploratory research notebooks. Treat them as prototypes unless the user makes one active.

## Data Conventions

- Current local Ti data generally resolves through `DATA(path=1, LAT=..., nnx=..., dis=..., dN=..., mechMode=...)`.
- `DATA(path=1, ...)` points into `Z:/p1/data/Ti/{dis}/{path_add}/{dN}/{LAT}/` and uses `PATH_PER` for `Z:/p1/data/Ti/per/{LAT}/`.
- `path_add` values such as `Target-xs` are part of the data tree. Keep them explicit in notebook configuration cells.
- Active lattice names include `FCC`, `tri`, `kagome`, and `hex`; older helpers also know about `square` and `45square`.
- Detailed transfer names, processed schemas, sample alignment, field metadata, and downstream consumers are maintained in the `review-p1-p2-data-contract` skill reference. Use it whenever those boundaries change.

- `validation_property_figures.py` owns the three production-style property
  figures under Ti Validation/audit/property-figures. Their captions/metadata
  retain calculation status and the length-scaled Hsieh reference. Original
  IDs 1-1000 supply spread; ID0 is separate. Recalculation still precedes paper use.
- Square/45square is retired from active Ortega validation: out-of-domain chi.
  `validation_orthotropic.py` retains rejection evidence; no plot CLI. ValConvPlots
  calls `validation_elastic_calibration.py --plot-verification`: published FEM
  markers and own FE checks remain distinct. Failed queue status stays unchanged.
- `validation_curve_processing.py` is opt-in, never an MLdata migration.
  --prepare-paired-examples freezes the random seed/IDs; Abaqus Python
  --extract-paired-examples reads native pairs; --paired-examples writes8galleries.
  Hsieh FT uses paired causal2fn, order2/DC (our stated reconstruction), keeping
  events native. Above-Nyquist cutoffs remain unapplied. No mask leakage or
  continuum-only check falsely labelled discrete-lattice validation.

## Shared Calculation Checks

- `test_validation_shared.py` verifies corrected Ortega's approximate isotropic
  geometry limit (about-.8% at a/W=.5), exact isotropic M and tensor-scale
  homogeneity. These mathematical checks do not validate a disorder constitutive
  model or the plastic eta factor. Source/evidence review lives in `audit/shared`.
- Orthotropic SiC replay must use source B20mm and assigned418197MPa,.162791,
  already mapped. Saved damage criterion is DUCTCRT; Ti LE=.1105 does not apply.
  Use source W/a/axes and native times, not current Geometry defaults or padding.
- Mesh refinement is a separate UAT/FT gate before PSC (168 retained model sets);
  PSC includes28UAT plus28FT sources. The Al/tri benchmark remains material-
  specific. Preview layouts are allowed; final Results integration stays pending.
- Current curve paths do not support NaN tails: distinguish field valid masks,
  curve targets and network inputs. Preserve observed prefixes per metric and
  migrate masks end-to-end before overwriting MLdata. C.3 shows normalised
  Mohr-diagnostic ratios; do not infer invariant M or CT k0 from those plots.

## Shared Helpers To Respect

- `resources/data_processing.py` owns conversion from raw `transfer/` files into processed p1 CSV/NPZ files and manifests.
- `resources/MLdata.py` owns `DATA`, ML-ready loading, property extraction, field loading, split construction, scaling, and save helpers.
- `resources/calculations.py` owns curve smoothing/parsing and UT/FT mechanical-property calculations.
- `resources/lattices.py` owns geometry, node counts, connectivity, effective properties, stiffness matrices, isotropy, and anisotropy calculations.
- `StiffnessMatrix` now shows the explicit physical-C prismatic reduction and
  separate fracture modulus M; `SIMresults` calls the third effective-property
  return `M_I_pe`. For measured C from modified beam inputs, declare the
  already-reduced state instead. Current FCC corrections do not rewrite stored
  population metrics or the separate48-case validation route. The source-axis
  loading-axis factor must be converted consistently before using crack-axis M.

## How To Work Here

- Before editing a notebook, identify whether the change belongs in the notebook cell or in a shared helper. Repeated data-processing logic usually belongs in `resources/`.
- Do not run whole notebooks blindly. Many cells assume `Z:/` data, OneDrive validation workbooks, Abaqus transfer files, or specific current working directories.
- Keep active notebooks compact and current. When a cell is replaced by a helper or a cleaner workflow, remove the old code path and clear stale outputs instead of keeping a fallback copy.
- Preserve processed CSV row/index alignment. Dropping failed outputs must keep matching input, output, frequency, and manifest records aligned.
- Keep the periodic reference row (`sim_id == 0`) intact unless the task is explicitly about removing or changing the reference.
- Do not silently change smoothing thresholds, fracture-index logic, stiffness fits, effective-property formulas, or outlier rules. These are scientific assumptions, not formatting details.
- The validation-only audit fixes tensile-criterion/deletion endpoints across filters;
  method-dependent peaks are secondary. Do not clip negative Apl, invent STATUS
  after termination, borrow a cutoff from another specimen, or promote a modulus
  sensitivity to paper truth. Zero-phase intact-prefix and through-deletion
  controls have different edge information and doubled-pass attenuation.
  Unit-load C0 is initial elastic compliance, not post-yield unloading compliance.
  Rectangular Explicit B21 surface degradation can precede whole-element
  deletion because section-point failure and whole-element deletion differ.
  An outer-surface strain criterion is a source-comparison choice; do not alter
  shared STATUS failure labels or damage laws to make them coincide.
  Force-only and matched U/RF causal filters remain labelled controls; matching
  their phases does not resolve the demonstrated hex80 negative work.
  Geometric elastic stiffening also makes that initial-compliance subtraction
  negative without plasticity. ALLSE comparisons remain diagnostic; do not
  substitute W-ALLSE for ASTM Jpl. Exact-source B21 C gives conversion 1/S11
  directly, without an additional plane-strain correction. Verified moduli and
  cutoffs apply only to their indexed source case, never neighbouring specimens.
  Shared calculation/export defaults and p2 products remain outside this route.
  `ValConvPlots` preserves source values but excludes incomplete saved FT rows and
  KJ<KI (negative-area) points. PSC criteria/sizes stay fixed and deferred.
- If changing field-output handling, preserve sample ids, frame values, node labels, node coordinates, components, valid masks, and main-body node filtering.
- Keep notebook configuration cells explicit for `LAT`, `nnx`, `dN`, `dis`, `path_add`, `mechMode`, and output type. Avoid hidden module-level state.

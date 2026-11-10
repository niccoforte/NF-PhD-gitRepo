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
- `validation_history_audit.py` reads indexed density-campaign ODBs without upgrade,
  retaining actual times, NaNs/validity, monitored STATUS and named ALLKE/ALLIE.
  `validation_modal.py` explicitly prepares/runs local exact-input eigenproblems
  and 1 N elastic-compliance controls; preparing/extracting is not submission.
  `validation_calculations.py` compares native/old-CSV moving averages, causal
  Butterworth and labelled sensitivities under `Validation/audit`; `--workbook`
  reconciles FTval read-only. The README owns commands and coverage.
- `validation_benchmark.py` digitises Hsieh Fig.10 with graphical bounds and an
  overlay; `validation_comparison.py` joins those points to exact saved/native
  identities. Only 3 mm is published in Fig.10; 10 mm scaling is an inference.
  `validation_diagnostics.py` prepares/extracts exact-source B21 periodic and
  nonlinear-work controls; `validation_queue.py` alone submits reviewed manifests
  serially, verifies delivery, then removes marker-owned scratch. Windows
  `launch` detaches a new reviewed queue; logs and serial-owner gates remain.
  `validation_workbook_fig10.mjs` authors the additive Deshpande mirror with
  Artifact Tool, backs up current bytes and verifies every existing package part
  and cell before replacement. It refuses occupied target rows; do not rerun it
  to overwrite an existing mirror. Canonical reference data remain the CSV.
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
- Publication views from `validation_comparison.py` use native initial-triangle
  results plus the explicitly adopted four-point supplement in
  `audit/new-validation/no-rerun-options-20261102.json`. Bind it to the selected
  CSV hash: two matched nonlinear-recovery indices and two density/length
  predictions retain distinct methods and case-specific captions, never invented
  native fracture KJ or source events. Native negative areas stay unavailable.
  Exports live in `audit/paper` as PNG/PDF/SVG with point/exclusion tables and
  captions. Fig.10 uses nominal relative density;3mm references are direct and
  10mm reference scaling is declared. Simplifying presentation must not change K.
  The separately named Fig.8b sqrt(3) hypothesis views may rescale displayed
  ordinates only with explicit per-point factors and captions; keep the base
  comparison unchanged and never promote either view to a verified correction.
  Use hollow reference outlines and layered simulation sizes without x jitter.
  Use exactly Hsieh et al. and Current study in the source legend, with common
  Current study circles. Identify topology beside the curves; retain internal
  Simulation series identifiers and all point-level calculation provenance.
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
- `validation_curve_processing.py` is an opt-in real-data filter/fit pilot,
  not a production migration. Hsieh FT quantities must use causal Butterworth
  P-v at2fn consistently; native events stay unfiltered. Order2/DC is a stated
  Abaqus-default reconstruction, not a reported Hsieh order. Masks cannot leak
  test-time termination. Continuum Ortega checks are not lattice FT validation.

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

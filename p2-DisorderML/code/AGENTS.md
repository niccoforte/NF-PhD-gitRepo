# p2 Notebook Agent Context

This directory is the local notebook layer for p2 training, diagnostics, tokenization, and exploration. Read the called `resources/` helper before changing notebook behavior. For active priorities, target runs, evidence, and the next handoff task, read `../PROJECT_STATUS.md`.

Dual/HPO notebooks use `Z:/p2` on Windows and ignored repo-root `data/` on Mac. The dual default retains the user's local `MULTI/Dual/Transformer/dual-MULTI-trial1` label and archived identity in metadata. Training/HPO are opt-in; saved results are not training data. Follow the root no-copy rule; never depend on ignored `samples/` code for execution or tests.

## Notebook Roles

- `review_damage_dataset.py` reviews explicit STATUS-only FIELDd products and execution metadata for the FCC dataset report. Selected preflight figures do not establish whole-corpus completion; preserve task/sample identities, unknown masks and complete binary legends. It opens no ODB and trains no model.

- `review_damage.py` validates explicit supported FCC UT/FT `strut_damage_v1` NPZs and builds a conclusions-first visual report without ODB access or training. Plot binary fracture from `strut_failed`/`strut_failure_known`: purple0 for known all-active, red1 for any verified failed member, black dashed for unknown. Do not colour active surface damage grey or use optional scalar severity as the fracture target. Raw SDEG is separate diagnostic context; track whether observed active surface damage later fractures within the saved window, never extrapolate beyond it. Use --ut/--ft for the periodic pair or repeated --case LABEL TASK NPZ for a labelled suite. Keep cases/IDs explicit, retain actual geometry and select FT zooms near the saved nominal tip. Report duplicate inputs/responses without treating filenames as independent designs. Audit-canonical endpoints are not saved ML node order; keep these products separate from displacement loaders.
- `ML-CurveOutputs.ipynb` and `ML-FieldOutputs.ipynb` are the main local curve- and field-output training/HPO notebooks.
- `ML-FieldToCurveOutputs.ipynb` is the exploratory field-input to curve-output notebook and should stay aligned with the HPC framework.
- `ML-DualOutputs.ipynb`: imports → Task → ML Model Name → DATA → HPO → MODEL → train/save/predict. Use direct `DUAL_DATA.from_files`, `DUAL_MODEL` and `run_dual_hpo` calls, not HPC subprocesses. `postprocess_load_dual_run` loads saved checkpoints; raw data loading is separate. Match the independent Outputs syntax; leave detailed plots to post-processing.
- `ML-DualPostProcessing.ipynb` follows inputs/config → Curve Models → Field Models → additional paired diagnostics. Reuse the existing curve/field diagnostic helpers through `load_dual_diagnostics`; never route dual checkpoints through legacy loaders. Feature generation and worked input examples belong in `samples/`, not these notebooks.
- `ML-CurvePostProcessing.ipynb` and `ML-FieldPostProcessing.ipynb` inspect one saved curve or field run respectively.
- `ML-HPOpostProcess.ipynb` retains curve/field model-specific and cross-model study sections; dual support uses `dual` / `MULTI` / `Transformer` / `model_hpo` and reviews all four winner outputs. No separate dual HPO notebook. Build full DATA explicitly in Outputs.
- `Tokenization.ipynb` follows the separate handoff in `../handoffs/TOKENIZATION_NEXT_STEPS.md`.
- `DimensionalityReduction.ipynb`, `GPR.ipynb`, `ML-DisorderDistribution.ipynb`, `Optimization.ipynb`, and `AK-ML-StressStrain.ipynb` are exploratory or historical unless the user makes one active.

## Notebook Boundaries

- Training notebooks train or tune models; single-run post-processing notebooks inspect saved runs; HPO comparison belongs in `ML-HPOpostProcess.ipynb`.
- Shared loading, diagnostics, and visualization logic belongs in `resources/MLdata.py` or `resources/MLmetrics.py` when more than one notebook needs it.
- Use saved `metrics.json`, diagnostic tables, `predictions.npz`, and `loss_history.csv` when available instead of rerunning a model merely for post-processing.
- Save post-processing figures under the run's `results/postProcessing/`, or `best_model_results/postProcessing/` for HPO best models. Do not create timestamped post-processing folders by default.
- The interactive field viewer is a required staple of field post-processing, alongside a static keyframe strip. Keep explicit sample/ID, frame and component controls; default FT maps to points because generic contour interpolation can bridge cracks. Maps use saved reference coordinates, not asserted mesh connectivity.
- Dual field subsections show both tasks with two live viewers; never close unrelated widgets. The final 2×2 compares UT field–curve, FT field–curve, UT–FT curves and UT–FT fields joined by sample ID. RMSE measures magnitude, not signed bias or unit-independent performance.
- Always plot losses on logarithmic axes. Keep raw task losses distinct from weighted contributions and physical diagnostic errors. HPO ranking is a fixed validation score, not its tunable weighted training objective.
- Dual HPO uses `HPO/<study>/best/results/`; point the dual review notebook at `best/`. Saved train-mean baseline scores remain authoritative when baseline arrays were not saved: do not replace them with validation means.
- Keep `Curve`, `Field`, and `FieldToCurve` behavior distinct. For names, shapes, metadata, layouts, and affected consumers, use `review-p1-p2-data-contract`.

## Training Conventions

- Outputs notebooks expose `FIELD_LOSS=None` to retain legacy behaviour, or an explicit dictionary of spatial/temporal/localisation weights. The shared `MLfield.field_loss_from_data` fits training-only scales. This does not change the existing HPO loss space. PostProcessing notebooks use saved motion/component diagnostics when present and clearly report absence for old runs; loss axes remain logarithmic. Do not equate motion-activity proxies with Abaqus failure labels.

- Geometry-to-curve uses `output_kind="curve"`; geometry-to-field uses `output_kind="field"`; field-to-curve uses `input_kind="field", output_kind="curve"` while saving under `FieldToCurve`.
- MLP consumes flattened curve inputs. Graph and Transformer workflows preserve node shape; field targets require node-compatible models, node-level output, and masked field loss.
- Keep model, loss, scaling, split, reduction, HPO budget, and geometry-feature choices explicit in notebook configuration.
- Do not silently broaden trial counts, epochs, datasets, model families, or scientific loss terms.

## Configuration And Editing

- Keep global configuration cells limited to durable run/loading choices such as root, task, model, run, split, device, load flags, and explicit path overrides.
- Put sample, frame, component, ranking, count, metric, and plot-style controls beside the visualization that uses them.
- Treat `None`, `""`, and `"auto"` consistently when a helper documents them as automatic path selection; normalize an override ending in `MLdata` before passing it to `DATA`, which appends that folder.
- Do not add module-level path constants or hidden notebook state to shared helpers. Prefer parameters and explicit configuration.
- Make broad notebook refactors interactively and cell by cell. Explain configuration changes, keep edits scoped to the agreed cell/helper, and ask before rewriting multiple sections.
- Remove replaced cells, stale outputs, obsolete imports, copied fallbacks, and abandoned helper paths from active notebooks.

## Validation

- Validate targeted notebook JSON and inspect changed cells; do not execute whole notebooks blindly when they require `Z:` data, saved runs, GPUs, or HPC.
- Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed` from the repository root, plus focused loader or diagnostic checks when the required artifacts are available.
- Record changing run observations or next steps in `../PROJECT_STATUS.md`, not here.

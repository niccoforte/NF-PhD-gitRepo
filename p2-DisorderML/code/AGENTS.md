# p2 Notebook Agent Context

This directory is the local notebook layer for p2 training, diagnostics, tokenization, and exploration. Read the called `resources/` helper before changing notebook behavior. For active priorities, target runs, evidence, and the next handoff task, read `../PROJECT_STATUS.md`.

## Notebook Roles

- `ML-CurveOutputs.ipynb` and `ML-FieldOutputs.ipynb` are the main local curve- and field-output training/HPO notebooks.
- `ML-FieldToCurveOutputs.ipynb` is the exploratory field-input to curve-output notebook and should stay aligned with the HPC framework.
- `ML-CurvePostProcessing.ipynb` and `ML-FieldPostProcessing.ipynb` inspect one saved curve or field run respectively.
- `ML-HPOpostProcess.ipynb` compares model-specific and cross-model HPO studies.
- `Tokenization.ipynb` follows the separate handoff in `TOKENIZATION_NEXT_STEPS.md`.
- `DimensionalityReduction.ipynb`, `GPR.ipynb`, `ML-DisorderDistribution.ipynb`, `Optimization.ipynb`, and `AK-ML-StressStrain.ipynb` are exploratory or historical unless the user makes one active.

## Notebook Boundaries

- Training notebooks train or tune models; single-run post-processing notebooks inspect saved runs; HPO comparison belongs in `ML-HPOpostProcess.ipynb`.
- Shared loading, diagnostics, and visualization logic belongs in `resources/MLdata.py` or `resources/MLmetrics.py` when more than one notebook needs it.
- Use saved `metrics.json`, diagnostic tables, `predictions.npz`, and `loss_history.csv` when available instead of rerunning a model merely for post-processing.
- Save post-processing figures under the run's `results/postProcessing/`, or `best_model_results/postProcessing/` for HPO best models. Do not create timestamped post-processing folders by default.
- Keep `Curve`, `Field`, and `FieldToCurve` behavior distinct. For names, shapes, metadata, layouts, and affected consumers, use `review-p1-p2-data-contract`.

## Training Conventions

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

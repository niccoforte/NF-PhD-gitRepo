# Field ML Post-Processing Context

This directory contains the ML notebooks and scripts for the disorder-to-response framework. The active task is keeping the field-output ML workflow clean, legible, and compatible after the post-processing refactor.

## Current Research Context

- The framework maps disordered 2D FCC lattice node coordinates to mechanical response.
- Curve-output models predict macroscopic stress-strain or force-displacement curves.
- Field-output models predict per-node displacement fields over Abaqus field-output frames.
- Current field target is displacement `U` only, usually components `U1` and `U2`.
- Field data is stored in final ML-ready `.npz` files named with the existing input/output convention, ending in `allFIELD.npz`.
- UT field outputs must use main-body nodes only. Grip-section nodes are dropped to match the input-processing convention.
- FT field output shape already looked correct and does not need the UT grip-node crop.

## Important Paths

- Local field ML data is expected at:
  `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`
- Local run archive root is expected at:
  `Z:/p2`
- Current downloaded field Transformer run:
  `Z:/p2/UT/Transformer/TR-Field-UT-1`
- Main notebook under active refactor:
  `p2-DisorderML/code/ML-FieldPostProcessing.ipynb`
- Main ML helper files:
  `resources/MLfunc.py`, `resources/MLmodels.py`, and `resources/MLmetrics.py`
- `resources/MLmetrics.py` owns curve/field diagnostics, diagnostic plotting, and saved-run post-processing helpers.
- `resources/MLfunc.py` should stay focused on ML training helpers, loss functions, HPO utilities, activation diagnostics, and older general ML plotting helpers.
- `resources/MLmodels.py` should stay focused on model classes and model lifecycle behavior such as training, evaluation, saving, and loading.

## User Priorities For This Refactor

The user wants a guided, cell-by-cell cleanup. Do not jump ahead and rewrite large sections without agreeing on the cell being changed.

Core visual goals:

1. Load all saved field outputs and choose sample, frame, and component directly in the visualization cell.
2. Support selected sample index, best sample, worst sample, and random sample views.
3. Make the field visualization interactive enough to justify using a notebook.
4. Show loss evolution over training for HPC runs.
5. Improve field maps beyond colored points, ideally with a toggle between point maps and continuous-style plots like the continuum plots.

Keep these goals active throughout the whole refactor.

## Configuration Cell Direction

The top field post-processing configuration cells should only contain durable run/loading settings:

- `RUN_ROOT`
- `mechMode`
- `model`
- `run_name`
- `VIEW_MODE`
- `run_type`
- `RUN_PATH_OVERRIDE`
- `LOAD_DATA`
- `LOAD_MODEL`
- `DEVICE`
- `DATA_PATH_OVERRIDE`
- `ACTIVE_SPLIT`

Viewing controls should not live in the global configuration cell:

- `FIELD_VIEW_FRAME`
- `FIELD_VIEW_COMPONENT`
- `FIELD_SELECTED_SAMPLES`
- `FIELD_RANDOM_COUNT`
- `FIELD_RANKING_METRIC`
- `FIELD_PLOT_STYLE`
- `NODE_METRIC`

Those belong in the relevant visualization cells.

## Helper-Code Rules

- Keep helper changes minimal and focused.
- Do not add module-level path constants or hidden global variables to helper scripts.
- Function parameters/defaults are acceptable when a default is needed.
- The default data path for post-processing should live as a function parameter/default, not as a module-level variable.
- `postprocess_load_data()` should treat `None`, `""`, and `"auto"` as using its default data path.
- If the path points directly to an `MLdata` folder, normalize it to the parent folder before passing to `DATA`, because `DATA` appends `MLdata`.
- Avoid stale helpers, stale imports, and old notebook cells as the refactor proceeds.
- Do not leave dated, legacy, stale, broken, duplicated, or unused code anywhere. Remove it when it is clearly obsolete, or move it into the module where it belongs if it is still active.
- Do not keep old notebook metadata snapshots or copied fallback versions of cells after a working implementation has replaced them.
- Prefer direct readable code over defensive safety-net layers unless the user has explicitly asked for that recovery path.

## Known Recent Fixes

- `postprocess_load_data()` in `resources/MLmetrics.py` now defaults to:
  `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`
- It normalizes an `MLdata` folder path to its parent before constructing `DATA`.
- `postprocess_build_diagnostics()` can now recover node coordinates from saved node-metrics CSVs when `DATA` is unavailable, using `x` and `y` columns.
- The current run has saved node metrics with coordinates, so field maps should not depend strictly on a loaded `DATA` object.

## Current Run Artifacts

For `Z:/p2/UT/Transformer/TR-Field-UT-1`:

- `predictions.npz` contains saved truth and prediction arrays.
- Current prediction and truth shapes observed earlier:
  `(889, 800, 40)`
- This corresponds to flattened field targets that are reshaped by diagnostics into sample/frame/node/component form.
- `model.json` has summary metrics, but not full loss history.
- Full training loss evolution was not saved for the current run except sparse values printed in the Slurm log.
- Future HPC runs should save loss history explicitly, for example as `loss_history.csv`, so post-processing can plot training and validation loss.

## Tone And Workflow

- The user wants to work through this interactively.
- Explain each configuration variable before changing it.
- Ask before making broad notebook edits.
- When editing, keep changes scoped to the agreed cell or helper.
- If a cell or helper is stale, point it out and propose removing it before doing so.

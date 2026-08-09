# p2 Project Status

Read this file for continuation, planning, or handoff work. Durable framework rules remain in `AGENTS.md`; Git history records completed changes.

## Current Objective

- Repository guidance identifies `code/ML-FieldPostProcessing.ipynb` as the current field-visualization/refactor focus; whether this remains the active priority is `To be confirmed`.
- The intended approach is guided, cell-by-cell cleanup rather than a broad notebook rewrite.

## Authoritative Working Surfaces

- Active notebook: `code/ML-FieldPostProcessing.ipynb` (`To be confirmed`).
- Shared loading and diagnostics: `resources/MLdata.py` and `resources/MLmetrics.py`.
- Saved-run behavior: `resources/MLmodels.py`; training/loss behavior: `resources/MLfunc.py`.
- Tokenization has its separate handoff in `code/TOKENIZATION_NEXT_STEPS.md` and should not be folded into the field-refactor task.

## Current Inputs And Evidence

- Expected local data: `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`.
- Previously documented run: `Z:/p2/UT/Field/Transformer/TR-Field-UT-1` (`To be confirmed` as the current run).
- That run was documented as having `predictions.npz` truth/prediction arrays of shape `(889, 800, 40)`, node-metric coordinates, and no full saved loss history. These observations require re-verification against the local run before use.
- Newer runs may contain `loss_history.csv`; post-processing should prefer it when present.

## Desired Reviewable Outcome

- Select sample and frame/component controls in the relevant visualization cell.
- Support selected, best, worst, and random sample views.
- Use saved metrics and coordinates without requiring model inference or a loaded dataset when those artifacts suffice.
- Keep figures under the run's `results/postProcessing/` directory and support point and continuous-style field views where the saved metadata permits them.

## Decisions And Next Task

- `Decision required`: confirm whether field post-processing is still the active p2 priority and whether `TR-Field-UT-1` is still the target run.
- Next reviewable task, if confirmed: inspect the current notebook and target run artifacts, then agree the first configuration or visualization cell to clean up.
- Full training, HPO, saved-run, `Z:`-data, and HPC validation has not been run during this guidance update.

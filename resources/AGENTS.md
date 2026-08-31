# Shared Resources Agent Context

`resources/` is the shared Python package for the paper folders. It is not a loose utilities directory: it defines the lattice geometry, Abaqus post-processing, p1 data products, p2 ML data loading, training, HPO, and diagnostics contracts.

## Module Map

- `imports.py` is a convenience import bundle used by older notebooks. Prefer explicit imports in new code.
- `lattices.py` owns `Geometry`, lattice dimensions, relative-density thickness, node counts, connectivity, effective properties, stiffness matrices, isotropy, and anisotropy helpers.
- `abaqus.py` owns Abaqus-facing helpers for node generation, disorder sampling, input exports, ODB history parsing, ODB field parsing, and old continuum displacement exports.
- `calculations.py` owns curve smoothing/parsing, UT/FT mechanical-property calculations, fracture geometry factors, and a legacy `FEA_run` hook.
- `data_processing.py` owns conversion from p1 raw `transfer/` files into processed input/output CSVs, manifests, field indexes, and stacked field NPZ files.
- `MLdata.py` owns the `DATA` class, path resolution, p1/p2 ML-ready data loading, property extraction, split construction, scaling, dimensionality reduction, node filtering, field loading, and MLdata saving.
- `MLmodels.py` owns model classes, `MODEL`, data loaders, train/predict/evaluate orchestration, checkpoint metadata, saved-run layout, and result artifact writing.
- `MLfunc.py` owns training loops, curve/field losses, HPO helpers, activation diagnostics, and older ML plotting helpers.
- `MLmetrics.py` owns saved-run loading, curve/field diagnostics, plotting, HPO summaries, and post-processing helpers.
- `tokenization.py` owns the output-informed tokenization prototype for recurring disorder motifs.
- `utilities.py` contains file renaming, Abaqus `.inp` editing, and backup-deletion helpers. Treat these as operational scripts, not general-purpose library functions.

## Pipeline Contract

`resources/` is the transformation boundary between p1 producers and p2 consumers. Use `review-p1-p2-data-contract` for its detailed file families, identity rules, schemas, field metadata, saved layouts, and impact map. Preserve the contract or migrate every affected consumer together.

## Abaqus Boundary

- `abaqus.py` wraps Abaqus imports in `try` blocks, but ODB/model/session functions still require Abaqus objects such as `openOdb`, `mdb`, `session`, and Abaqus constants at runtime.
- Text-only helpers such as input-file parsing can be inspected in standard Python, but do not assume ODB behavior has been validated without Abaqus.
- Do not introduce standard-Python dependencies into Abaqus-critical code unless they are available through `requirements-abaqus.txt` or the setup scripts.
- Keep `resources.abaqus` imports from breaking normal notebooks where possible, but validate Abaqus behavior in the correct interpreter when making substantive changes.

## Lattice And Mechanics Rules

- `Geometry` supports lattice families such as `FCC`, `square`, `45square`, `tri`, `kagome`, and `hex`. Many formulas depend on exact `nnx`, `nny`, `L`, `H`, `W`, `ai`, `vol`, `totalNodes`, and `totalBracketNodes`.
- If changing a lattice definition, update node counts, connectivity assumptions, fracture crack positions, stiffness calculations, and field body-node masks together.
- `calcUT` and `calcFT` encode current scientific definitions for ductility, strength, stiffness, work of fracture, fracture force/displacement, and toughness metrics. Do not adjust thresholds, smoothing, or fitted regions without documenting why.
- `calc_FaW_aniso`, `calcC_mohr`, `calcC_sims`, and anisotropy helpers are used by validation/stiffness workflows. Keep units and plane-strain assumptions explicit.

## Data-Processing Rules

- `data_processing.py` reads raw files from `Path(dat.PATH) / "transfer"` and periodic references from `dat.PATH_PER` when available.
- Processed curve CSVs are written to `dat.PATH` and aligned by integer simulation id.
- Manifest CSVs record missing inputs, missing outputs, frequency handling, NaNs, failure-index drops, and final inclusion. Preserve this audit trail when changing filtering.
- Field metadata, array layout, and body-node filtering are contract details maintained in the data-contract skill reference. Do not change them without tracing p1 exporters through p2 diagnostics.

## ML Data And Model Rules

- `DATA(path=0, ...)` is the legacy Akash-data path; `DATA(path=1, ...)` resolves local `Z:/p1/data/Ti/...`; `DATA(path="HPC", ...)` resolves the cluster p2 data root; explicit paths are accepted.
- `DATA` appends or expects `MLdata` depending on context. Post-processing helpers normalize paths that already point to an `MLdata` folder.
- Curve models may use flattened or node-shaped inputs. Field models must preserve node structure and use node-compatible models such as GNN/GCN/GAT/Transformer.
- Do not apply input PCA/reduction before node tokenization or graph/Transformer node-shape workflows.
- `MODEL.save()` and `MODEL.save_results()` produce checkpoint, JSON metadata, prediction, metric, loss-history, and diagnostics artifacts consumed by `MLmetrics.py`.
- HPO helpers save Optuna studies and best-model artifacts in model-specific or cross-model layouts. Keep these layouts stable unless all loaders are updated.

## Editing Guidance

- Avoid adding paper-specific assumptions to shared functions when a parameter or notebook/script configuration is enough.
- Prefer small, named helper functions over copying data-processing code across notebooks.
- Keep shared helpers compact and purposeful. Remove workaround helpers, fallback branches, and compatibility scaffolding once they no longer serve a concrete current workflow.
- If a helper is used once and does not clarify the main flow, consider inlining it. If logic is reused or makes notebooks cleaner, keep it in the appropriate `resources` module.
- Preserve backward compatibility for saved artifacts where practical; old runs are research evidence.
- When a change touches p1 and p2, validate at the lowest common contract: file names, sample ids, array shapes, metadata keys, and loader behavior.
- Treat `utilities.py` operations that rename files, edit `.inp` files, or delete `.bak` files as destructive. Use dry-run paths when available.

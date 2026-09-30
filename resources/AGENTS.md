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
- `MLdual.py` owns the opt-in joint UT/FT serial Transformer path. Its `DUAL_DATA` adapter reuses existing `DATA` products; its two Transformer stages share parameters across tasks and train through one scalar objective/optimizer without changing legacy behavior.
- `MLdualHPO.py` owns the opt-in fixed-score search, conditional ranges, study lock, scratch/archive synchronization and study-level resume. Keep search implementation out of notebooks and preserve the single-run default training behaviour.
- `MLfunc.py` owns training loops, curve/field losses, HPO helpers, activation diagnostics, and older ML plotting helpers.
- `MLfield.py` owns opt-in signed displacement-difference losses, physical jump scales and specimen-specific localisation weights. Automatic edges use the existing periodic FCC connectivity and initial FT cut; other lattices require validated explicit edges.
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
- `DUAL_MODEL` checkpoints use their own versioned descriptor and do not pass through legacy `MODEL` reload logic. Keep this separation until every dual saved-run consumer is explicitly implemented.
- The dual `fcc_ti` context profile validates the canonical FCC grid and FT subset. Reference x0/y0/designable are shared; presence, UT body interfaces, FT pin/coupling and nominal-tip features are task-specific. Pin membership uses each sample's initial disordered coordinates before scaling. Never label coupled body nodes as directly fixed/prescribed DOFs. Use the same context helpers in the data adapter, reports and notebooks.
- Dual saved results reuse `MLmetrics` field/curve diagnostics in physical units. Preserve canonical prediction arrays and explicit validity masks; exclude absent task nodes from field diagnostics. Pass inverse-affine curves to `CombinedCurveLoss` using Torch buffers so physical curve features and joint gradients remain meaningful; normalized MSE remains an explicit alternative.
- `load_dual_diagnostics` adapts one saved task to existing visualizers and retains saved train-baseline scores. `plot_dual_loss_history` always uses log axes. Additional HPO selection metrics, epoch callbacks, gradient clipping and per-stage learning-rate groups are opt-in; old dual checkpoints/defaults remain loadable.
- `postprocess_load_dual_run` discovers single-run or HPO `best/` artifacts and optionally reconstructs the dual checkpoint for inspection/inference, reusing saved loss builders. Attaching DUAL_DATA requires matching sample IDs, normalizers, node masks and physical context. This is not training/optimizer resume. `plot_dual_sample_errors` joins all four sample tables by unique simulation ID. Field viewers must coexist without globally closing widgets.
- The curve dashboard omits ratio-line points with near-zero true variation (relative threshold 1e-6, absolute floor 1e-12). This is display-only; never alter saved diversity metrics to make a plot look better.
- HPO helpers save Optuna studies and best-model artifacts in model-specific or cross-model layouts. Keep these layouts stable unless all loaders are updated.

## Editing Guidance

- Dual architecture experiments are opt-in: `private_layers` separates only field encoder blocks; input projections and the curve stage remain shared. `LocalFieldGraph` uses one shared message MLP, task-specific initial edges and differentiably reconstructed initial disordered coordinates, with both message directions and degree averaging. No new node tokens, hard local-only attention or evolving-damage assumptions. Preserve old shared-stage state-dict names/defaults.
- `residual_fields=True` fits per-node/frame/component mean and variation on training only, with an explicit pooled-scale floor; the saved affine inverse provides the fixed mean skip. It changes both the field target coordinates and their identical curve-input coordinates. `detach_fields` is a separate gradient-routing diagnostic; `true_curve_weight` adds true-field supervision to the same curve network and is recorded in training metadata.
- `curve_only_source=true|predicted` freezes the field stage (including dropout), zeroes field loss weights and trains only a fresh shared curve stage. Save/restore this trainer setting; source-conditioned inference requires DUAL_DATA. Field artifacts always remain frozen predictions, never substituted perfect targets. Use two-curve fixed selection for these fits; the default four-output HPO score is unchanged. Save both true/predicted input comparisons and distinguish an oracle from deployable predictions.
- `StructuredFieldLoss.fixed_weights` optionally supplies positive a-priori node/frame/component weights, normalised over each specimen's valid entries. Defaults and historical state dictionaries remain unchanged. `fcc_initial_crack_region` validates the reference FCC grid and converts A1's dN=.2 CrRegMESH box into a 125-node retained FT prior; it is not the Abaqus edge set or future fracture path. Current target-activity localisation is still not affine-invariant anomaly detection.
- Legacy GNN `graph_semantics="fcc_initial_v1"` explicitly selects the validated FCC initial cut and bidirectional message edges. `historical` remains the default and old-descriptor fallback. Never silently change historical graph checkpoint interpretation. Setup signatures distinguish the new semantics.
- Corrected legacy GNN training rejects padded 800-node FT data (its pooling has no presence mask); use native 788-node DATA. The graph helper supports both indexing layouts, while the dual graph route supports masked canonical padding.
- `dual_design_diagnostics` uses validation only: UT own-curve post-peak 1% work, FT full-domain work explicitly labelled a proxy, objective ranking/top recovery/regret, optional explicit strength threshold. It is not a toughness calculator, Pareto search or training objective. Keep physical cutoff decisions and old HPO selection unchanged.
- Curve-source comparison plots prefer explicit predicted-field evaluation tables. A `curve_true` fit's standard curves are oracle predictions, not predicted-field inference; never compare those against themselves under two different labels.

- Optional `crack_face=True` in DUAL_DATA factories appends `initial_crack_face` using the existing `reference_field_edges` UT/FT degree difference, gated by FT presence (26 nodes; UT=0). Preserve the default 11-channel schema and saved checkpoints; opt-in adds one channel for fresh models. The nominal tip lies on the intact strut joining (120,90) and (120,100), not a node. Do not confuse this static feature with evolving damage or attention connectivity.

- Structured field losses preserve legacy defaults. Invert affine field scaling in Torch before spatial/temporal differences, fit jump scales on training targets only, mask both endpoints/frames, and node-average incident edge errors. Target-dependent capped weights are training-only and must not encode a presumed fracture region. Retain separate raw component logs and a fixed checkpoint-selection metric across ablations.
- `field_motion_diagnostics` reports displacement activity, not verified damage. Timing and principal-axis metrics use explicit heuristic eligibility thresholds and must retain event counts/ambiguity. FT tip-region reporting is diagnostic only. Frozen independent curve comparisons use ID-aligned validation intersections, the original curve input scaler and unchanged static features; dual true-field substitution reports masked/imputed entries. Do not alter a live HPO objective when adding these diagnostics.

- Avoid adding paper-specific assumptions to shared functions when a parameter or notebook/script configuration is enough.
- Prefer small, named helper functions over copying data-processing code across notebooks.
- Keep shared helpers compact and purposeful. Remove workaround helpers, fallback branches, and compatibility scaffolding once they no longer serve a concrete current workflow.
- If a helper is used once and does not clarify the main flow, consider inlining it. If logic is reused or makes notebooks cleaner, keep it in the appropriate `resources` module.
- Preserve backward compatibility for saved artifacts where practical; old runs are research evidence.
- When a change touches p1 and p2, validate at the lowest common contract: file names, sample ids, array shapes, metadata keys, and loader behavior.
- Treat `utilities.py` operations that rename files, edit `.inp` files, or delete `.bak` files as destructive. Use dry-run paths when available.

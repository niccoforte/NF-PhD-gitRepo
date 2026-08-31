# p2 Project Status

Read this file for continuation, planning, or handoff work. Durable framework rules remain in `AGENTS.md`; Git history records completed changes.

## Current Objective

- Build and independently verify an end-to-end design route from the full FCC nodal-disorder field to paired tensile and fracture performance.
- The intended surrogate has one nodal disorder/geometry input and two load-case branches linked by the same disorder sample. Each branch is serial: geometry -> predicted displacement-field history -> predicted macroscopic response curve -> derived mechanical properties.
- Use the frozen surrogate in a constrained multi-objective optimisation (MOO) workflow to estimate the attainable Pareto set, then verify predeclared extreme and knee/compromise candidates through fresh paired FEA.

## Authoritative Working Surfaces

- Geometry-to-curve training and diagnostics: `code/ML-CurveOutputs.ipynb` and `code/ML-CurvePostProcessing.ipynb`.
- Geometry-to-field training and diagnostics: `code/ML-FieldOutputs.ipynb`, `code/ML-FieldPostProcessing.ipynb`, and `code/ML-HPOpostProcess.ipynb`.
- Field-to-curve development: `code/ML-FieldToCurveOutputs.ipynb` and `HPC/FieldToCurve/`.
- Optimisation prototypes: `code/Optimization.ipynb`; this is not yet a production full-dimensional FCC MOO implementation.
- Shared loading and diagnostics: `resources/MLdata.py` and `resources/MLmetrics.py`.
- Saved-run behavior: `resources/MLmodels.py`; training/loss behavior: `resources/MLfunc.py`.
- Tokenisation remains a preliminary future design-space direction with a separate handoff in `code/TOKENIZATION_NEXT_STEPS.md`; it is not part of the core current MOO implementation.

## Current Scientific Decisions

- Optimise the complete 722-node by two-degree-of-freedom disorder representation: 1,444 continuous design variables.
- Keep generated candidates within the training-domain disorder bound corresponding to 20% of the shortest strut length. The production generator must be checked to freeze whether this is a coordinate-wise or radial bound.
- Use one shared geometry across aligned UT and FT/C(T) branches.
- Current recommended primary objectives are UT work to failure and FT `K_JIC`, with tensile ductility and crack-initiation displacement reported separately. The exact tensile objective remains subject to supervisor confirmation.
- Retain tensile strength through a predeclared data-derived constraint based on the strongest strength-retaining, high-performance training samples. The operational threshold must be fixed before final optimisation.
- Treat optimisation output as an estimated finite-search Pareto set. Any reported meeting point is a selected knee/compromise candidate, not a unique or global optimum.
- Require fresh paired FEA before claiming that a generated candidate improves performance or remains Pareto dominant.

## Current Inputs And Evidence

- Expected local data: `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`.
- The authoritative data drive is temporarily inaccessible at the QM Engineering building following a fire. Do not infer final dataset counts or rerun data-dependent studies from partial local notebook outputs while access is unavailable.
- Available saved notebook outputs are validation/HPO evidence rather than untouched final-test evidence.
- Direct geometry-to-UT-curve MLP validation currently performs approximately at the mean-curve baseline and exhibits severe response-diversity collapse; it is not optimisation-ready.
- Direct geometry/disorder-to-curve attempts have not learned the relationship adequately. The displacement field is therefore retained as the learned intermediary rather than using a parallel curve readout from the geometry latent state.
- UT field-to-curve Transformer validation using true FEA fields is the strongest current curve result, but it does not establish end-to-end geometry-to-response performance.
- Geometry-to-field Transformer validation is currently modest for UT and stronger for FT; both require frozen-manifest locked-test evaluation.
- The two serial stages currently exist as separate workflows; the unified one-input, UT/FT dual-output model is not implemented.
- The principal unresolved training problem is simultaneous supervision at different depths:

  `L_total = sum_m(lambda_field,m * L_field,m + lambda_curve,m * L_curve,m)`, for `m in {UT, FT}`.

  The field loss directly supervises the disorder-to-field stage, while the curve loss supervises the field-to-curve stage and backpropagates through both Transformers. `MaskedFieldMSELoss` is the current pointwise field baseline and needs further development; `CombinedCurveLoss` is the current full-curve objective. The weighting/balancing strategy is not yet fixed.
- Exact generated, completed, failed, filtered and paired sample counts conflict across historical sources and must be resolved from recovered manifests.

## Evidence Required Before Full Manuscript Claims

- Freeze authoritative sample and split manifests, preserving a genuinely untouched test set.
- Complete fair direct geometry-to-curve baselines and both geometry-to-field-to-curve chains, including FT field-to-curve modelling.
- Develop and predeclare the field-specific loss and field/curve/UT/FT weighting strategy for unified end-to-end training.
- Evaluate chained inference with predicted rather than true FEA fields and quantify propagated error in optimisation-relevant metrics.
- Select and implement the production 1,444-variable constrained MOO algorithm with matched-budget simple-search baselines, repeated seeds, convergence diagnostics, geometric feasibility, and uncertainty/out-of-distribution safeguards.
- Re-simulate a predeclared set of Pareto candidates under paired UT and C(T) FEA and recompute dominance from FEA outputs.
- Keep tokenisation outside the principal manuscript result unless held-out enrichment, stability, and motif intervention tests are completed.

## Decisions And Next Task

- `Decision required`: confirm with supervisors whether UT work to failure, ductility, or both define the tensile optimisation objective; freeze the strength-threshold rule; and select the primary MOO strategy.
- While the data drive remains inaccessible, keep the unified architecture and loss formulation at the presentation/documentation stage unless the user explicitly resumes code work. Synthetic/toy interface checks may be planned separately without claiming scientific validation.
- Once data access is restored, the first task is to inventory and freeze the paired manifests and split IDs before any final retraining, test evaluation, or optimisation run.
- Full training, locked-test evaluation, full-dimensional MOO, fresh FEA verification, `Z:`-data validation, and HPC validation have not been run during this status update.

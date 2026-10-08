# Physics and strain supervision — feasibility task

Location: p2-DisorderML/handoffs/. Source paths below retain their repository/p2 scope.

This is a proposed dedicated continuation, not a dispatched chat or permission
for bulk exports, new FEA or model changes. Read applicable repository guidance
and current PROJECT_STATUS. Reconcile existing code before acting. Accuracy owns
the active displacement losses and sudden-motion weighting; coordinate changes
to MLfield/MLdual/MLmetrics rather than introducing another trainer.

## Why separate this work

Validating mechanics residuals for a fracturing beam lattice is iterative.
Displacement targets contain translations, not beam rotations, stresses,
plastic state or damage history. A simple axial-spring equilibrium residual
would not represent the actual bending/plastic/fracture simulation. Do not
label any target-derived loss a PINN merely because it uses derivatives.

Current spatial and temporal losses match signed displacement differences to
truth after differentiable inverse normalization. They remain active options.
The proposed sudden-motion weights combine changes in temporal increments and
local non-affine motion, not displacement magnitude or an assumed crack path.
They are kinematic proxies, not verified damage labels. Do not duplicate that work.

## First test: displacement-derived finite strain

Inspect p1/code/ContinuumPlots.ipynb and its actual coordinate conventions.
It constructs initial-connectivity triangles and calculates
F = dx @ inverse(dX), E = (F.T @ F - I)/2. Later plotting interpolation is
separate and may bridge cracks. Reuse these definitions, not an unrelated
small-strain/Delaunay implementation.

Start with a small read-only true-field audit and exact numerical tests:
translation and finite rigid rotation give zero E; known affine stretch/shear
gives the analytic signed tensor; a local opening produces large apparent E.
Check initial disordered coordinates, triangle orientation, conditioning,
initial FT crack exclusion and validity at all three nodes/frames. Document
that apparent continuum strain across a subsequently broken strut is an opening
indicator, not intact-material strain. Never mask using unknown future damage.

If that audit supports it, propose one opt-in auxiliary loss computed from
predicted/true nodal displacements in Torch. Keep nodal outputs and field-to-curve
bridge unchanged; no triangle tokens/extra head needed. Retain displacement loss,
signed Exx/Eyy/Exy (not only magnitude), train-only scale and bounded influence.
Compare matched displacement-only versus displacement+strain, same split/seed,
fixed physical validation ranking; report field, increment, localization and
downstream curve errors. Audit computational overhead before production.
Strain-only prediction changes the output contract and does not uniquely recover
rigid motion/displacement; it is a later experiment, not the first step.
No plastic-strain export is requested.

## Second test: actual FT coupling kinematics

Inspect A1's coupling DOFs/reference-point histories and original named sets.
Fit/check whether true coupled-node trajectories satisfy the allowed common
reference-point motion and rotation. Use finite-rotation kinematics where needed.
Do NOT label coupled body nodes individually fixed or set all their U values
to zero. UT interfaces are not the outer constrained grip nodes.
The adapter sees body nodes, not the full FE mesh: establish what can actually
be constrained using available outputs. If true fields do not satisfy the
proposed residual within export accuracy, stop and explain rather than train
against an incorrect law. Only then propose a differentiable residual and
isolated loss ablation. This is the first plausible physics-residual candidate,
not a validated existing PINN term.

## Do not implement by default

- Full momentum balance without rotations, internal forces, plastic/damage
  history and the appropriate quasi-static/dynamic interpretation.
- Per-node monotonic displacement, fixed diagonal shear bands, symmetry for
  individual disordered specimens, or gradients forced toward zero.
- Force/energy reconstruction, ODB damage extraction or design optimisation.

Deliver labelled numerical examples and feasibility verdicts in p2/samples.
Keep notebook sources in p2/code, results in data, existing checkpoints/defaults
unchanged. Any later HPC test must use B1, a small GPU gate and fixed matched
comparisons, with explicit user authority in that dedicated chat.

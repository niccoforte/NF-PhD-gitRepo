# SimCheck task checklist

Unique checklist for chat01a1782e-8158-7792-90e7-6fd570dd3155.
Updated10November2026. [Current evidence and figures](README.md).
This replaces accumulated superseded proposals; it is the single task register.

## Accepted scope and decisions

- [x] Consolidate useful Ti validation sources, recover native histories and
  reconcile the20 Review lattices.py findings with verified foundation fixes.
- [x] Close the Hsieh comparison display:14 native initial-triangle points,
  2 measured nonlinear-recovery indices and2 predictions, with provenance.
  Direct3mm Fig10 is primary;10mm references are declared length-scaled values.
  No further full-fracture completion campaign is selected. No clipping negative Apl.
- [x] Retain Appendix C quasi-isotropy approximation for tri/kagome/hex and
  orthotropic FCC. Its20-realisation study is a sensitivity basis, not proof of
  unchanged modulus or exact isotropy for every ID. No per-realisation tensor
  programme is selected. The blanket24000-job proposal is withdrawn.
- [x] Complete4 actual UAT initial-tangent jobs and4 three-step body-tensor
  diagnostic jobs. They are new linear Abaqus FE, not analytical values or
  original-curve fits. UAT Eapp hex1.574/1.528GPa,kagome7.448/5.625GPa.
- [x] Record measured short-job costs and population counts in
  [runtime-budget.json](Z:/p1/sims/Ti/Validation/audit/shared/runtime-budget.json).
  UAT10-11s,CT7-16s,three-step tensor10-11s;8CPUs serial. No population queue.
  Missing517UAT values:~2-3h queue plan. All1973hexUAT:~6-12h if needed.
  Counts7877UT/7911FT,7788paired IDs; original scopeIDs1-1000,ID0 reference only.
- [x] Verify corrected Ortega approximate isotropic ASTM limit and exact M limit.
- [x] Recover213 actual published FEM markers from Ortega2014Fig7, with source
  hash/vector coordinates. Current equation differs at most1.602% graphically.
  Separately18 current-study C(T) jobs/9 tensors verify elastic f/M within
  .883%K/1.773%J. Never label designed own-FE tests as published measurements.
- [x] Resolve loading-axis Ortega versus crack-axis Bao convention: paired f/M
  or fB=fO*sqrt(MB/MO). Shared FCC calculation corrected; equal-axis outputs stable.
- [ ] Verify the actual FCC beam-lattice elastic conversion. The18 continuum
  CPS4R jobs and published Ortega FEM points are not end-to-end lattice-fracture
  validation. Start with3 actual orderedFCC B21 C(T) elastic models at a,a+h,
  a+2h. Compare full-opening compliance energy over the same crack interval with
  the interval-average Ortega KI^2/M; test h/2h sensitivity. No published matching
  planarFCC initiation-toughness dataset is currently identified. Do not claim
  this energy check independently validates a local fracture criterion.
- [x] Retire square/45square from active Ortega validation: out-of-domain chi.
  Remove plot CLI/derived displays and replace notebook caller. Original source
  files and rejection receipts stay only pending final cleanup disposition.
- [x] Produce native hex-ID1 threshold/cutoff figure. Raw6.9271%,filtered7.6708%,
  previous moving average~7.6709%. A10.736% difference is not proof filtering is
  wrong. Hsieh uses real-time Butterworth at2fn before mechanical calculations;
  our.4Nyquist second-order forward/backward trial is not that calibration.
- [x] Apply manuscript figure conventions: tri red,kagome black,hex green,FCC
  blue; full orange,mechanism blue; varepsilon_L/Sigma; no Hsieh statistical bars.

## Shared calculations before population replacement

- [x] Adopt Hsieh's filtered mechanical response for FT performance calculations:
  causal Butterworth,fc=2fn from the matched specimen's lowest physical mode.
  Filtered endpoint load and filtered P-v work feed KI,Apl,J,KJ consistently.
  Order2/DC is an explicit Abaqus-default reconstruction choice; the paper does
  not state the order. Moving averages and.4Nyquist forward/backward examples
  are diagnostics, not the selected primary method. Material events stay native.
- [ ] Apply that protocol to matched native histories and regenerate validation
  values/plots before final replacement. Reuse periodic modal controls, with a
  bounded disorder check; no per-realisation modal programme. Saved-history
  filtering cannot recreate unavailable integration-level data or undo aliasing.
- [ ] Settle one scalar UAT stiffness rule from resolved early response; use
  exact-source linear FE only where unresolved. Do not average incompatible
  hex fits or use periodic E to erase disorder softening. The517NaN repair
  alone does not validate all finite historical stiffness values. Give a bounded
  cost before submitting any selected replacement population queue.
- [ ] Settle C(T) k0 extraction from reusable early histories; use short source
  controls only where needed. Appendix C symmetry does not itself give cracked-
  specimen compliance. No blanket all-ID control programme is selected.
- [ ] Settle the scalar isotropic conversion and representative FCC M under
  the same material-state convention as the retained controls. Reuse Appendix C
  evidence; no automatic full FCC tensor scaling from one UAT slope, no second
  plane-strain correction, and no full tensor solve for every disordered ID.
- [ ] Consolidate one final StiffMatrix workflow, retaining only controls used
  by the accepted calculations. Preserve distinct material tensor versus
  cracked-specimen compliance roles; retire duplicated unused implementations.
- [ ] Complete orthotropic/FCC plastic calibration after the lattice elastic
  check. Initial feasibility cap:6 orderedFCC beam models,3 admissible crack
  lengths under original incremental plasticity and3 under a potential-based
  nonlinear material fitted only to the source constituent response. Damage off,
  stationary notches and monotonic loading through the existing initiation range.
  Verify constitutive/local/global response equivalence and the elastic limit
  before an independent fixed-displacement energy derivative is used as Jref.
  Then evaluate eta=B*b0/Apl*(Jref-KI^2/M) over resolved positive Apl and held-out
  loading levels; further crack-increment/mesh checks only if the pilot passes.
  Six runs are feasibility, not completed validation. No independent orthotropic
  plastic result exists yet. Do not equate ALLIE/ALLPD/unloading differences or
  frozen-state deletion work with J. Time the first nonlinear pilot; the elastic
  10s job timing is not a nonlinear runtime estimate. No pilot jobs launched here.
- [ ] Replace exporter/event/filter/work/stiffness/modulus paths in place with
  the settled implementation; no parallel legacy calculation options. Run a
  representative end-to-end pilot before any population recalculation.

## Convergence and remaining validation

- [ ] Check UAT and FT mesh convergence first, reusing168 retained model sets
  and MeshConv source data. Check the recalculated quantities and charts.
- [ ] Recalculate both28UAT and28FT PSC cases with final methods; keep the
  supervisor-approved specimen sizes and convergence criteria.
- [ ] Light Al/triangular UAT/UTval processing and plot check. Existing7ODBs,
  6transfers; re-export18-1.1 if needed. No new nonlinear campaign selected.
- [ ] Revisit lattice-foundation branches only where active later work uses them.

## Deferred ML migration and population processing

- [ ] Preserve201common planned-time/loading positions. Current producer stores
  one periodic x row and y-only disorder rows. Never stretch a short prefix,
  rescale every specimen endpoint or adopt the illustrative0-10% query crop.
  Native support determines masks; actual paired coordinates remain for physics.
- [ ] Implement finite placeholders plus masks through scaling/loaders/losses/
  derivatives/areas/metrics. Use per-specimen valid mean then specimen mean,
  preventing automatic count-based downweighting of short histories. Missing
  targets do not enter scaling/loss; masks do not leak future response to inputs.
- [ ] Preserve25%ductility/1%work thresholds; non-crossings stay censored, not
  invented failure indices. Keep metric-specific validity without losing useful
  remaining properties. Compare qualified denoising under matched folds/seeds
  and truncation tests before changing ML targets or an existing model winner.
- [ ] Recalculate originalIDs1-1000 per topology/design with exact UT/FT pairing,
  periodicID0 separate. Verify inclusion after stiffness repair; no row imputation.
- [ ] Replace Excel/notebook values, population tables/statistics and figures
  together after acceptance; remove obsolete calculation cells/audit variants.

## Manuscript and final cleanup

- [ ] Coordinate one Ortega-based methods section and appendix showing published
  elastic verification, isotropic ASTM comparison and completed plastic check.
- [ ] Move UAT/FT validation into Results with periodic markers and disorder
  population ranges. Include final FCC FT, without inventing a Hsieh FCC reference.
  Use median/IQR and5-95% ranges with counts; label as population spread.
- [ ] Keep the Al triangular UAT benchmark material-specific, expand Ti results
  to all4topologies, and include periodic ductility markers/guides and disorder
  ranges using final values. Select final layout after numerical recalculation.
- [ ] Replace manuscript numerical claims/figures only after final population
  outputs are accepted. Corrections owns direct manuscript editing.
- [ ] Verify consumers, then remove superseded code/results, auxiliary solver
  bulk, duplicated exports and all named task scratch. The
  [cleanup register](Z:/p1/sims/Ti/Validation/audit/cleanup-review.json) owns paths;
  preserve unique source evidence until its explicit final disposition is applied.

Author-closed side investigations are removed from this scope; existing evidence
is not deleted or described as numerically negligible by that decision.

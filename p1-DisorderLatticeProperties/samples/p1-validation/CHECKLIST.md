# SimCheck task checklist

Unique checklist for chat01a1782e-8158-7792-90e7-6fd570dd3155.
Updated10November2026. [Current evidence and figures](README.md).
This replaces accumulated superseded proposals; it is the single task register.

## Accepted scope and decisions

- [x] Consolidate useful Ti validation sources, recover native histories and
  reconcile the20 Review lattices.py findings with verified foundation fixes.
- [x] Recalculate source-matched Hsieh filtering and replace current views:
  13available native initial-triangle points,1displayed recovery index and
  2declared predictions. Available native values take precedence; hex10mm16/20%
  filtered areas are negative and remain excluded. Before-state is fingerprinted.
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
  The517NaNs are516hex-full and hex-mechanismID737; exactIDs in
  audit/shared/uat-stiffness-missing.csv. Old endpoint int(.1*peakindex) can
  leave only2samples and deliberately returnNaN. Finite values are not thereby
  wrong: kagomefullID1 differs.074% from its exact linear solve. No517/1973
  rerun programme is selected. Prior counts are restrictedIDs1-1000 snapshots;
  newerFCC extensions are not silently added to the agreed population scope.
- [x] Verify corrected Ortega approximate isotropic ASTM limit and exact M limit.
- [x] Recover213 actual published FEM markers from Ortega2014Fig7, with source
  hash/vector coordinates. Current equation differs at most1.602% graphically.
  Separately18 current-study C(T) jobs/9 tensors verify elastic f/M within
  .883%K/1.773%J. Never label designed own-FE tests as published measurements.
- [x] Resolve loading-axis Ortega versus crack-axis Bao convention: paired f/M
  or fB=fO*sqrt(MB/MO). Shared FCC calculation corrected; equal-axis outputs stable.
- [x] Quantify actual FCC beam-lattice elastic conversion with3 exact-source
  B21 crack lengths a80/90/100mm and one source-CT periodic mapped tensor.
  M8945.725MPa,lambda1,chi.590409. Interval-matched Ortega energy differs
  by9.48-12.28%J, equivalent4.63-5.96%K; h/2h comparison retained. This is
  numerical lattice verification, not independent local-failure validation.
- [x] Find a published in-domain orthotropic lattice benchmark: Omidi/St-Pierre
  2023 snub-square, lambda1,chi7.65/4.74/3.16, actualCTa/W=.25. Current
  Ortega factors differ3.51-6.75% from published TableC2 factors.
- [ ] Reproduce the3 published snub-square beam-fracture points if retained as
  the end-to-end literature validation. New topology needed; do not relabel the
  retired square/45square or continuum verification as these beam results.
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
- [x] Apply the protocol to matched native ISO histories with15 verified
  periodic sources. Previously available14:12filtered,2sampling-unresolved;
  hex3mm16% becomes positive while hex10mm16/20% become negative. Available
  native count14to13. All16events and31diagnostic/QS/recovery rows unchanged.
  No cutoff capping, missing-frequency reconstruction or MLdata rewrite.
- [ ] Resolve the reported work-partition/finite-sampling qualification before
  production population replacement; filtering alone did not remove negative Apl.
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
- [ ] Complete orthotropic/FCC plastic calibration. All6 source-mesh nonlinear
  jobs and4 elastic/tensor controls completed. Final conditional eta1.825/2.482/
  2.180 depends on elastic subtraction versus ASTM2.261. Material/local unloading
  and load/crack-stencil sensitivity prevent a qualified original-model factor;
  do not promote the closest variant. An independently established original-law
  discrete energy reference remains necessary. No arbitrary extra queue is
  selected; preserve the full numerical sensitivity and source/delivery hashes.
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

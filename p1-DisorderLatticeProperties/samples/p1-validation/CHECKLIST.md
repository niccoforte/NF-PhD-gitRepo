# SimCheck task checklist

Unique checklist for chat01a1782e-8158-7792-90e7-6fd570dd3155.
Updated 23 November 2026. [Current evidence and figures](README.md).
This replaces accumulated superseded proposals; it is the single task register.

## Accepted scope and decisions

- [x] Consolidate useful Ti validation sources, recover native histories and
  reconcile the20 Review lattices.py findings with verified foundation fixes.
- [x] Preserve the approved Figure 10 axes, labels, titles and legend in the
  23 November fixed-cutoff update. Figure 8 remains frozen. Changes to other
  final figures require explicit approval.
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
- [ ] Reconcile the external Omidi lattice benchmark before calling it validated.
  All3 periodic beam tensors and all submitted CT/BL diagnostics are extracted.
  PolymerCT20 failed atLE=.064 before.1 initiation;25/30 unsubmitted. ElasticCT
  global stress is pin-controlled, rejected. Released-notch-phase109l BL gives
  -4.1/-11.2/-19.3% against Table3's elastic fit; Fig2c-centred phase has larger
  residuals despite the expected vertical critical bar. Do not choose the phase
  that gives closer agreement. Source code/paper differences and corrected
  angular branches are recorded. Prepared larger/refined checks are unsubmitted;
  no further solver campaign until this discrepancy is understood.
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

- [x] Approve the fixed topology-by-test Butterworth cutoffs, the paired causal
  order-two implementation, and native material events. Use the exact
  [registry](../../code/response_filter_cutoffs.json). See [the filtering
  rationale](../../FILTERING.md) for the rounded table, scope, sweep thresholds,
  examples, and independent checks. The earlier modal archive cutoff and
  solver-output trials remain baseline comparisons in the evidence report.
- [x] Recalculate and update the final Hsieh Figure 10 comparison with the
  approved C(T) registry. The 16 native histories pass sampling and event
  transfer checks. The display contains 14 filtered native points and the
  disclosed hexagonal 10 mm/20% raw fallback, 3 mm/20% recovery estimate,
  and two 13% predictions. Five cases exceed at least one archive-baseline
  response tolerance. The underlying table preserves signed areas, native
  event brackets, and the sensitive 3 mm/20% filtered candidate.
  [Transfer checks](Z:/p1/sims/Ti/Validation/audit/paper/figure10-transfer-checks.csv)
- [x] Apply the registry to the frozen 64-history population pilot. All response
  curves and 40%-work fits resolve. Sixteen short histories keep their original
  position masks. Native section fields support 31 of 32 C(T) endpoints.
  Triangular full ID 661 records damage before the strain criterion. Hexagonal
  mechanism FT ID 336 has negative plastic work in the curve-stiffness check.
  The 1% UAT work endpoint is censored for hex full ID 826 and FCC mechanism
  ID 717. [Pilot report](Z:/p1/sims/Ti/Validation/audit/curve-processing/accepted-filter-study/fixed-cutoff-pilot/pilot.json)
- [ ] Reprocess the full population after the remaining shared-calculation and
  convergence checks. Requalify hexagonal UAT curve stiffness at 300 Hz, keep
  the nine source-exact elastic fallbacks and transfer the native-event and
  censoring rules before replacing saved results.
- [ ] Once all checks and population values are final, SimCheck passes the
  manuscript work to Corrections. Update the methods reference and relevant
  appendix with the approved cutoff table, qualification and its limitations,
  and raw/filtered examples for periodic and disordered UAT and C(T). Apply
  final population and figure changes in that manuscript pass.
- [x] Implement one curve-derived initial-work slope through40% filtered peak,
  with30/50% windows, cutoff sensitivity and energy-curvature screens. All64
  sampled histories pass. Against4 UT controls errors-.33/+3.34%; against11
  FT controls-4.72/+1.94%. This estimates scalar specimen response, not C.
- [x] Diagnose and repair the two-point guard: all1973 old hex UT estimates
  finite,1456 already-finite values and other properties unchanged. No reruns.
- [x] Complete9 source-exact elastic holdouts:6 within5%, full21 +7.47%,
  mechanism698 -6.39%,699 -6.71%. All numerical/hash checks pass. The accepted
  1973-ID catalogue retains1964 qualified curves and uses all9 preflagged elastic
  controls uniformly, not just the3 larger errors. Flag membership was frozen
  before FE; the fallback policy was selected after reviewing the holdouts.
  Original predictions remain unchanged. No more hex stiffness simulations.
  All9 raw/filtered curves,40% work lines,30/50% intervals and per-ID CSVs are in
  accepted-filter-study/flagged-hex-curves. Every original flag is solely >5%
  window spread, not missing data, a failed simulation or a cutoff failure.
- [ ] Resolve negative-area cases/finite-sampling limits during the production
  pilot; filtering alone is not a repair for an invalid initial-triangle area.
- [ ] Recompute saved property columns and audit restored UT/FT pair membership
  during the agreed population pass. No Props/MLdata rewrite has been performed.
- [x] Qualify reusable-history C(T) k0 using the same initial-work rule and11
  exact-source1N controls. No blanket disordered-compliance simulation programme.
- [ ] Settle the scalar isotropic conversion and representative FCC M under
  the same material-state convention as the retained controls. Reuse Appendix C
  evidence; no automatic full FCC tensor scaling from one UAT slope, no second
  plane-strain correction, and no full tensor solve for every disordered ID.
- [ ] Consolidate one final StiffMatrix workflow, retaining only controls used
  by the accepted calculations. Preserve distinct material tensor versus
  cracked-specimen compliance roles; retire duplicated unused implementations.
- [x] Process 15 phase-matched original-law isotropic C(T) jobs. Triangular
  median load eta is 2.2663 and work eta is 2.1220 against ASTM 2.2610. Kagome
  gives 2.1428 and 2.0116 against ASTM 2.2593 at 66% common plastic-opening
  coverage. Hexagonal leaves no resolved monotonic plastic branch. These are
  pre-initiation checks. [Isotropic results](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/isotropic/load-separation/eta-results-r2.json)
- [x] Complete the three-notch Kagome extension with the exact R2 1 N
  controls. The common range reaches the observed LE11 initiation bracket.
  At the last intact opening, load/work eta are 2.339/2.162 against ASTM
  2.259, changing equivalent K by +0.65%/-0.80%. Compliance and fit-window
  sensitivity limit independent factor identification. [Kagome results](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/isotropic/load-separation/kagome-bracket/eta-results.json)
- [x] Replay the ordered FCC size series at the approved 225 Hz C(T) cutoff.
  Ortega plus prescribed ASTM eta gives equivalent K differences of +3.76%,
  +1.83%, +0.57% against the scaled beam reference for nx20, 30, 40. The
  retained original-law FCC load-separation estimates at the longer cap are
  1.874 and 1.728. Their saved E11 field gives no independent LE11 initiation
  bracket. Retain eta=2.261 as a working total-J approximation for the tested
  series. [FCC checks](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/README.md)
- [x] Complete bounded FCC same-interval total-J verification at the resolved
  initiation opening. Use independent potential-energy differences, prescribed
  ASTM eta2.261, original-flow transfer and matching elastic baseline. At common
  observed opening6.45203mm (99.871% of event), original-flow K errors4.02-4.68%.
  inc_a0 has a reviewed converged prefix; inc_a2 solved fully but packaging failed.
  Preserve both failure records. No fitted eta is promoted or additional rerun
  selected. This verifies the adopted approximation, not an exact original-flow
  contour integral or a published FCC elastoplastic comparison.
- [ ] If the manuscript claims independent FCC plastic-factor validation,
  establish that separately: the completed total-J/potential approximation
  supports the combined near-initiation calculation, not unique identification
  of eta or the original-flow plastic term. Retain prescribed2.261 meanwhile.
  A conditional near-initiation potential-reference fit gives eta1.804787
  (-20.18% from2.261);90-100mm held-out interval K residual at most1.972% for
  the potential law,2.576% for original-flow transfer. It is not a production
  coefficient or independent original-flow validation; its elastic residual,
  material approximation and shared90mm control remain explicit.
  Original-law long-strip pilot is under audit/fcc-calibration/original-flow-strip.
  H=90 mm failed at remote-edge cutbacks. The H=190 mm cracked strip converged
  through a genuine onset bracket before later cutbacks. Its applied J bracket
  is 128.265–134.476 N/mm. The 0.3044% force imbalance exceeds the 0.1% gate,
  and the matched C(T) local fields and domain/mesh evidence do not qualify
  transfer. [Failed-prefix review](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/original-flow-strip/collection-h190far/failed-prefix-review.json)
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

## Manuscript integration

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
Cleanup is deferred and tracked separately in [CLEANUP-CHECKLIST.md](CLEANUP-CHECKLIST.md).
Do not remove audit evidence, worktrees or temporary files before that review.

Author-closed side investigations are removed from this scope; existing evidence
is not deleted or described as numerically negligible by that decision.

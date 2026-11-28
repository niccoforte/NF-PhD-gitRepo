# SimCheck task checklist

Unique checklist for chat01a1782e-8158-7792-90e7-6fd570dd3155.
Updated 25 November 2026. [Current evidence and figures](README.md).
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
- [x] Defer Omidi reproduction for the ordered FCC closeout. Retain the published
  Table C.2 elastic comparison as supporting evidence. The unresolved source
  versus paper notch phase and pin-controlled or incomplete C(T) diagnostics are
  recorded in the audit. A future external reproduction claim requires its own
  phase, boundary, mesh and initiation checks.
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
- [x] Requalify the 1,973 saved hexagonal UAT transfer histories at 300 Hz.
  Keep all nine frozen source-exact elastic fallbacks. Of the remaining curve
  estimates, 1,942 pass and 22 are newly window-sensitive (5.02–5.83%).
  All estimates are finite. The transfer replay uses the declared time schedule.
  [Evidence](Z:/p1/sims/Ti/Validation/audit/production/hex-uat-cutoff-requalification.json)
- [ ] Transfer native-event and censoring rules before population replacement.
  All 8,000 disorder IDs have paired INP/ODB files in the archive by filename.
  Check geometry identity, observed support and metric eligibility at replay.
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
- [x] Verify the negative-area case in the frozen pilot. Hexagonal C(T)
  mechanism 336 gives -32,799 Nmm with curve stiffness and -27,308 Nmm
  with a source-exact 1 N control. Withhold total J/KJ and retain elastic K.
  [Compliance diagnostic](Z:/p1/sims/Ti/Validation/audit/production/hex-ct336/diagnostic-result.json)
- [ ] Carry the negative-work and unresolved-event exclusions through
  convergence and population processing. Keep valid properties from a
  partly supported run. A recovered-work estimate uses a different partition.
- [x] Reopen the two withheld C(T) archives. Triangular full 661 damages in
  compression away from the notch before the 50–55 ms notch bracket.
  Hexagonal mechanism 336 still gives negative signed work at a valid
  20–25 ms notch bracket. Exclude both C(T) toughness values while retaining
  eligible UAT responses. [Case dispositions](Z:/p1/sims/Ti/Validation/audit/production/pilot-exception-disposition.json)
- [x] Compare Mohr and Abaqus B21 on matched mapped-material periodic patches.
  Young's modulus differs by less than 0.5% for triangular, kagome and FCC.
  Axial-affine hexagonal overpredicts by a factor of 6.33.
  [Elastic model comparison](Z:/p1/sims/Ti/Validation/audit/production/disorder-conversion-appendix/periodic-elasticity-comparison.json)
- [x] Revise the disorder conversion appendix. Use one point per FCC geometry
  and report all four topology sensitivities. Reference Appendix C before
  the conversion appendix in the fracture methods. Use periodic a0 for
  disorder and retain the realised-notch check only as sensitivity evidence.
  [Appendix draft](Z:/p1/sims/Ti/Validation/audit/production/disorder-conversion-appendix/README.md)
- [x] Measure exact main-body elastic tangents for all periodic UAT specimens.
  Retain the main-body gauge for the corrected UAT properties.
  [Gauge definition and comparison](Z:/p1/sims/Ti/Validation/audit/controls/periodic-uat-body-histories/README.md)
- [x] Finish the four periodic main-body UAT replays. All four contain the
  initial modulus window and the ductility and work endpoints. Retain the
  native and archived-clamp comparisons separately.
  [Periodic gauge checks](Z:/p1/sims/Ti/Validation/audit/controls/periodic-uat-body-histories/README.md)
- [x] Apply the narrowest gross main-body section to kagome UAT stress,
  using the left and right outer beam centrelines of each specimen.
  Retain full gross width for hexagonal. Matched periodic and ungripped
  hexagonal B21 models agree within 0.015%. The fixture raises the exact
  elastic slope by about 29%.
  [Stress-area and fixture checks](Z:/p1/sims/Ti/Validation/audit/controls/periodic-uat-body-histories/README.md)
- [ ] Requalify the main-body gauge on disordered UAT records. Match the
  UAT gauge in both terms of the paired kagome modulus ratio before the
  population recalculation.
- [ ] Recompute saved property columns and audit restored UT/FT pair membership
  during the agreed population pass. No Props/MLdata rewrite has been performed.
- [x] Qualify reusable-history C(T) k0 using the same initial-work rule and11
  exact-source1N controls. No blanket disordered-compliance simulation programme.
- [x] Measure the four ordered 10 mm/20% conversion references in the
  [material catalogue](Z:/p1/sims/Ti/Validation/audit/controls/population-materials/material-catalogue.json).
  Triangular, kagome and hexagonal mapped B21 values are 9.250, 9.079 and
  1.404 GPa. The hexagonal isotropy residual is retained. FCC uses a physical
  periodic tensor and one prismatic reduction, giving 8.340 GPa.
- [x] Screen ordered versus specimen-specific conversion on the frozen paired
  population pilot. The UAT slope ratio changes KJ by at most 0.84% for
  triangular and 0.54% for hexagonal, and lowers it by 0.50–4.83% for
  kagome. Both FCC sensitivity models use the same axes and one reduction.
  The Mohr tensor changes KJ by at most 0.06%, and the bending-aware model
  gives up to 1.41% lower KJ. Replacing the specimen C(T) initial slope
  with the periodic slope has a larger effect, up to +13.53% for kagome.
  [Per-specimen sensitivity evidence](Z:/p1/sims/Ti/Validation/audit/production/disorder-conversion-sensitivity.json)
- [x] Settle the disorder conversion for initiation KJ. Retain the ordered
  triangular, hexagonal and FCC conversions. Use the paired UAT apparent
  stiffness ratio for kagome. The expanded FCC screen includes 30 full and
  30 mechanism-based geometries, crossed with eight observed plastic fractions.
  The bending-aware KJ sensitivity reaches 1.45% and 0.38% respectively,
  with eta held at 2.261. [Appendix draft and evidence](Z:/p1/sims/Ti/Validation/audit/production/disorder-conversion-appendix/README.md)
- [x] Verify the approved rules on all 64 native pilot histories. Save the
  specimen C(T) initial stiffness and eligibility separately. Retain the
  nominal notch in the calculation and record the realised centreline notch
  for geometry checks. The pilot has 30 eligible C(T) toughness values.
  [Updated pilot](Z:/p1/sims/Ti/Validation/audit/production/population-properties-pilot-kagome-area.json)
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
- [x] Process the original triangular three-notch initiation bracket. Load/work
  eta are 2.294/2.172 against ASTM 2.261, with full observed overlap.
  [Original triangular fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/isotropic/load-separation/tri-bracket/eta-results.json)
- [x] Process the larger ordered isotropic cases at their original A1 notches.
  Load/work eta are 2.190/2.059 for kagome 30, 2.180/2.019 for kagome 36,
  and 2.287/2.147 for triangular 40. All three reach full observed overlap.
  The triangular reference reached the strain bracket before a later solver
  failure. The verified initiation prefix and failure record are retained.
  [Larger-size results](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/supervisor-20261123/report.md)
- [x] Diagnose the hexagonal signed plastic work from the exact R2 ODBs. At the
  longest notch, Abaqus reports +16,713 Nmm plastic dissipation despite a
  -588 Nmm initial-triangle area. The archived elastic and unloading controls
  explain why repeating the same work partition cannot identify ASTM eta.
  [Energy receipt](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/isotropic/load-separation/hex-energy-check.json)
- [x] Complete the ordered FCC initiation fit at the original 80 mm notch
  of the 20×19 specimen. The stabilised reference reaches the observed strain
  bracket and the retained 90/100 mm companions give full overlap. Load/work
  eta are 1.871/1.786. Treat 1.871 as a calibration candidate. With Ortega, the
  resulting J is 128.88 versus 128.29 N/mm from the screened beam fields
  (+0.46% J, +0.23% equivalent K).
  [Initiation fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-le-bracket/a0-rerun/stabilization-1e-5/eta-results.json)
- [x] Replay the ordered FCC size series at the approved 225 Hz C(T) cutoff.
  With Ortega and fixed eta 1.871, matched-ring equivalent K differences are
  +0.23%, −0.72%, and −1.34% at nx20/30/40. The wider ring sensitivity makes
  the larger sizes diagnostic. Raw histories and alternate elastic subtraction
  give different trends. Direct load-fit convergence remains unmeasured.
  [FCC size evidence](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/approved-cutoff-check.json)
- [x] Complete bounded FCC same-interval total-J verification at the resolved
  initiation opening. Use independent potential-energy differences, prescribed
  ASTM eta2.261, original-flow transfer and matching elastic baseline. At common
  observed opening6.45203mm (99.871% of event), original-flow K errors4.02-4.68%.
  inc_a0 has a reviewed converged prefix; inc_a2 solved fully but packaging failed.
  Preserve both failure records. This earlier comparison tests the prescribed
  factor against the potential-energy approximation near initiation. The
  potential-material fitted coefficient stays a diagnostic.
- [x] Replay `eta=2.261` using the production FCC material state and source
  notch. At matched force, work and event, equivalent K differences against
  ordered beam fields are +6.06%, +4.40%, +3.32% for 20, 30 and 40 cells.
  [Shared mechanics pilot](Z:/p1/sims/Ti/Validation/audit/production/shared-mechanics-pilot.json)
  also records the canonical archive's distinct 20-cell initiation result.
  Hand off the size-sensitivity evidence and independent toughness comparison
  only after the author's explicit approval. The inverse-square guide fixes its asymptote to ASTM, and the
  limiting value remains unresolved.
  The 40-cell set yields load/work fits of 2.176/2.085 at the original notch.
  The 180 mm companion resolved initiation before the late failed tail, and the
  fit has full reference coverage. The matched-interval load secant rises by
  8.8% at the same normalized plastic opening. The 30-cell set gives
  2.363/2.298, so the direct size series is nonmonotonic. The 26/36-cell
  sets use original notches of 108/148 mm and share the serial archive queue.
  The 26-cell original-notch load/work fit is 2.177/2.144 at full initiation
  coverage. The 36-cell fit is 2.650/2.385 with full coverage, flagged for
  0.119% stabilisation energy at the 158 mm companion's fitted opening.
  The isolated 158 mm retry with finer increments and lighter stabilisation
  reached full coverage and gave 2.703/2.452. The 0.166% stabilisation energy
  at the fitted opening fails the 0.100% check. Retain the 36-cell flag.
  [Retry evidence](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n36/middle-notch-refined/fit-comparison.json)
  Its outer-notch load secant is 2.185. The 50-cell fit is 2.209/2.075 at full
  coverage, including both verified companion histories before late cutbacks.
  At common normalized plastic opening 0.003574, local load fits for 40/50
  cells are 2.140/2.209. The outer-pair load estimates move in the opposite
  direction across different notch intervals. Hold the 46/56-cell submissions
  because more sizes at the current notch intervals would leave the observed
  estimator and compliance sensitivity unresolved.
  [40-cell fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n40/eta-results.json)
  · [30-cell fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n30/eta-results.json)
  · [26-cell fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n26/eta-results.json)
  · [36-cell fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n36/eta-results.json)
  · [50-cell fit](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n50/eta-results.json)
  Representative disordered checks are needed for a population-wide claim.
  The potential-material fit and H190 strip bracket remain supporting checks.
  The strip's force imbalance and local-field mismatch leave C(T) transfer
  unresolved. The separate hexagonal work-partition qualification stays open.
  [Scope and limitations](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/supervisor-20261123/report.md)
- [ ] Replace exporter/event/filter/work/stiffness/modulus paths in place with
  the settled implementation; no parallel legacy calculation options. Run a
  representative end-to-end pilot before any population recalculation.

## Convergence and remaining validation

- [x] Reprocess all 168 UAT/C(T) mesh histories with source gauges and the
  native-event rules. UAT modulus is stable against the finer meshes.
  Kagome, hexagonal and FCC failure measures retain mesh sensitivity.
  For C(T), Kagome C5/F2 was recovered by aligning timestamps. The hexagonal
  results include a qualified archived fallback and half-rate C5/F15.
  The full-edge FCC C5 branch has all seven plotted points. Retain the
  event-specific source calculations in the comparison receipt.
  UAT figures report peak stress,
  initial modulus and dimensionless ductility using the manuscript symbol.
  The C(T) mesh figure includes initiation load-line displacement.
  In UAT PSC, the archived body height accounts for most of the difference
  from the historical ductility values for kagome and hexagonal. A shape-
  preserving guide follows the measured ductility points.
  [Mesh checks and figures](Z:/p1/sims/Ti/Validation/audit/production/convergence/README.md)
- [ ] Reconcile Appendix B with the active FCC C(T) coarse/fine mesh and
  qualify metric-specific convergence claims. Archived mesh metadata remain
  available in the case-level source receipts.
- [x] Reprocess all 28 UAT and 28 C(T) PSC archives with the source geometry
  and working ordered conversion. Use raw UAT histories for peak stress,
  stiffness and ductility. The C(T) figure displays all 24 values and
  the receipt records the event criterion per case. Both PSC figures
  stop at 36 cells.
  [PSC results and figures](Z:/p1/sims/Ti/Validation/audit/production/convergence/README.md)
- [x] Check the three negative hexagonal PSC areas using exact 1 N controls.
  The native-event areas remain negative at 26, 36 and 40 cells.
  [Compliance sensitivity](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-compliance/diagnostic.json)
- [x] Process the two matched 40-cell hexagonal PSC C(T) replays. Approved
  filtered work remains negative at both observed events. Raw values agree
  at the same opening after the rate change. Use the three reviewed archive
  raw fallbacks at the original events and keep their signed filtered results.
  [Rate and fallback comparison](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-psc40-rate/comparison.json)
- [x] Process the half-rate dense replay for hexagonal mesh C5/F15. It gives
  KJ = 9.498 MPa√m with the approved filter, with positive plastic work
  across all tested stiffness windows at both sampling rates.
  [Half-rate receipt](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-mesh-c5f15-rate/comparison.json)
- [x] Process PSC Hex 36 dense half-rate replay. Use the Butterworth
  result at the native 40 kHz endpoint, KJ = 9.087 MPa√m. Both exact
  1 N and the three slope windows give positive work on the dense signal.
  At 2 kHz the exact-compliance area is slightly negative.
  [PSC 36 receipt](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-psc36-rate/comparison.json)
- [x] Process hexagonal mesh C5/F20 at half rate. The unfiltered endpoint
  gives KJ = 9.261 MPa√m. Its work remains positive at both sampling rates
  with the exact 1 N control and the new response's 30%, 40% and 50% fits.
  [Comparison](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-mesh-c5f20-rate/comparison.json)
- [x] Qualify hexagonal mesh C1/F1. Two dense rate replays bracket the
  notch event and give negative work with the measured 5.908 kN/mm stiffness.
  The load–unload diagnostic confirms excessive elastic-work subtraction.
  Retain the observed load and opening. J and KJ remain unavailable
  under the common calculation, giving 83 of 84 toughness values.
  [Dense rate comparison](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-mesh-c1f1-dense-rate/comparison.json) ·
  [Load–unload diagnostic](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-mesh-c1f1-dense-rate/load-unload-diagnostic.json)
- [x] Verify the four PSC C(T) archives with earlier remote compression:
  FCC 16 and triangular 10, 16 and 20. Plot the recorded energy equivalents
  and preserve the observed elements, times and coordinates in the receipt.
  [Remote-damage evidence](Z:/p1/sims/Ti/Validation/audit/production/convergence/fcc-rear-rotation/first-damage-comparison.json)
- [x] Test rear-edge rotations on the FCC C5/F5 and C2/F5 meshes. Reject the
  tested restraint because C5/F5 still damages remotely and earlier, and C2/F5
  toughness changes by 3.11%. Record the first-damage load, opening and
  equivalent toughness for all eight archived remote initiation cases.
  [Boundary-condition comparison](Z:/p1/sims/Ti/Validation/audit/production/convergence/fcc-rear-rotation/first-damage-comparison.json)
- [x] Compare the four completed C5 rear-band controls. All still had first
  damage away from the notch. The uncompleted band jobs were stopped or
  superseded by the full-edge suite.
  [Band comparisons](Z:/p1/sims/Ti/Validation/audit/production/convergence/fcc-c5-rear-band-rotation/partial-comparison.json)
- [x] Process all seven C5 full-edge reruns. The zero-rotation condition
  is verified at all 96 right-edge mesh nodes. Use one full-edge C5 series
  in the mesh figure. Keep the C2 results and the archived C5 evidence.
  [Comparison](Z:/p1/sims/Ti/Validation/audit/production/convergence/fcc-c5-full-rear-rotation/partial-comparison.json)
- [x] Read all seven Al/triangular UAT histories, including the missing
  18-1.1 curve. Use the agreed 176 mm and 414 mm² nominal dimensions for
  the comparison figure, with the filtered trace through the first collapse.
  The peak is 16.344 MPa, 0.74% below Gu's digitised experiment. The legend
  sits inside the axes.
  [Al validation check](Z:/p1/sims/Ti/Validation/audit/production/convergence/al-validation.json)
- [ ] Integrate the agreed Al figure into Results after handoff approval.
  Retain its nominal benchmark convention and reference provenance.
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

- [ ] Give Corrections the approved Mohr–B21 paragraph and table for placement
  in Appendix C immediately after the stiffness-matrix methods. Cite Appendix C
  before the new disorder-conversion appendix in the fracture methods.
  [Appendix C insertion](Z:/p1/sims/Ti/Validation/audit/production/disorder-conversion-appendix/README.md#proposed-insertion-in-appendix-c)
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
For reproduction, use `validation_convergence.py extract --study mesh|psc|al`
in Abaqus Python, then `replay --study mesh|psc|al` in standard Python.
For the current mesh and PSC figures, pass `--raw-fallback-evidence` with the
[reviewed comparison](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-psc40-rate/comparison.json).
Pass `--rate-comparison` once each for the mesh
[C5/F15](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-mesh-c5f15-rate/comparison.json)
and [C5/F20](Z:/p1/sims/Ti/Validation/audit/production/convergence/hex-mesh-c5f20-rate/comparison.json)
receipts, and `--mesh-fcc-full-edge-evidence` with the completed C5 comparison. For PSC,
pass `--psc-first-damage-evidence` with the remote-damage evidence above and
`--rate-comparison` with the PSC 36 dense-rate receipt.
The Al overlay uses `validation_al.py` with the replay and original workbook.

Cleanup is deferred and tracked separately in [CLEANUP-CHECKLIST.md](CLEANUP-CHECKLIST.md).
Do not remove audit evidence, worktrees or temporary files before that review.

Author-closed side investigations are removed from this scope; existing evidence
is not deleted or described as numerically negligible by that decision.

# Damage Modeling -> Improve ML Accuracy: HPC continuation

From **Damage Modeling** (`01a13c9f-c95e-7ff1-854f-a3b2f4fbea6c`) to the current
Windows **Improve ML Accuracy** continuation (`bc645215-a354-451c-93f7-540e7d36d3dc`).
The researcher requests this handback for HPC tests, merged into that chat's
current workflow. The bounded local label audit is complete. Improve ML Accuracy
owns the next cluster checks and any separately agreed model integration.

Read applicable repository instructions, p2 PROJECT_STATUS, the relevant HPC run
guide/skills and the implementation below. This brief replaces the original
start-the-local-audit instructions; do not repeat the audit or revive old jobs.

## Merge with the current workflow

- Keep Case1(i): independent UT/FT models and separate field/curve training.
  `independent-inputs-261015` was submitted with gate31869463 and production
  31869464-31869478. These are historical submission IDs, not a fresh queue-state
  claim. Recheck its manifest/queue/results and preserve immutable snapshots;
  do not duplicate or alter the input study.
- Custom-loss submission remains ON HOLD until the researcher reviews the input
  winner. Inherit that winning training-input policy, including true/mixed or
  correction if selected; deployment remains predicted/corrected. Do not launch
  the current 30-job raw-P loss draft unchanged or add damage to it silently.
- Add damage to the existing Accuracy checklist as a distinct follow-on. Bounded
  execution/data-alignment checks can run while the input study proceeds. Place
  a later predictive damage ablation explicitly in the selected baseline's plan,
  preserving the held loss-study decision and one-change comparisons.
- This handback authorizes the receiving chat to progress HPC testing; it does
  not define a new damage architecture/loss, grant archive-scale extraction, or
  make the selected extreme specimens a training corpus. No jobs were submitted
  by Damage Modeling.

## Code and artifact locations

Implementation baseline: **7166de8**, verified on both Git remotes. This handoff's
own documentation commit follows it. The receiving Windows chat last used
`Z:/00-PhD-gitRepo`, clean at d3db4e8 when inspected; it needs a fast-forward to
the current remote, preserving any newer/unrelated work.

The actual local audit implementation and ignored products are in
`C:/Users/exy053/Documents/00-PhD-gitRepo`. Git carries code, not these ignored
research products. Read their reports directly on this host or transfer only the
explicit required files with a hash manifest before HPC use. Do not recreate a
Windows repo-root data/ folder. Raw sources are under
`Z:/p1/data/Ti/disNodes/0.2/FCC/sims/`.

| Product under p2-DisorderML/samples | Raw source stem | Final failed parents / elements |
| --- | --- | --- |
| damage-audit/UT-periodic-damage.npz | Ductile-FCC-20-per-1 | 83 / 83 |
| damage-audit/FT-periodic-damage.npz | Fracture-FCC-20-per-1 | 7 / 7 |
| damage-disordered/UT-397-maxD.npz | Ductile-FCC-20-20disNodes-lhs-all-397-maxD | 95 / 97 |
| damage-disordered/UT-747-maxB.npz | Ductile-FCC-20-20disNodes-lhs-all-747-maxB | 92 / 93 |
| damage-disordered/FT-747-maxB.npz | Fracture-FCC-20-20disNodes-lhs-all-747-maxB | 9 / 9 |
| damage-disordered/FT-747-maxK.npz | Fracture-FCC-20-20disNodes-lhs-all-747-maxK | 9 / 9 |

Each samples directory has a visual README, labelled figures, provenance and
independent verification JSONs. Source/code fingerprints are retained per file.
D means researcher-selected highest UT ductility, K initial FT toughness, B the
same-disorder UT/FT property-Pareto pair; this audit does not recalculate optima.

The FT maxB/maxK INPs and all exported nonmetadata arrays are identical, despite
different ODB bytes. Six files are five unique task responses across three
underlying designs, with only two complete UT/FT geometry pairs. They are audit
fixtures, not independent population/train/validation/test samples.

Executable sources:
- p1 `SIMscripts/A2_DamageOUTpostProcess.py`: one explicit ODB/INP, Abaqus Python,
  readOnly=True; no discovery, upgrades or job submission.
- `resources/damage.py`: checked original-joint identities, actual FE chains,
  missing-aware failure masks and strut_damage_v1 validation; pure NumPy.
- p1 `SIMscripts/test_damage_contract.py` and `test_damage_odb.py`: focused
  synthetic checks and independent live-ODB comparison respectively.
- p2 `code/review_damage.py`: saved-NPZ visual review, repeated --case arguments.

The quick start and array specification remain in
`p1-DisorderLatticeProperties/SIMscripts/DAMAGE.md`. Samples contains outputs,
not executable dependencies. No DATA/MODEL damage loader, head or training flag
has been implemented.

## Label and alignment contract

- Binary fracture is **1 if any member STATUS=0**, **0 if all member statuses
  are known active**, unknown otherwise. Use `strut_failed` under
  `strut_failure_known`; element equivalents retain original FE identity.
  Missing output is never filled with zero or interpreted as failure.
- SDEG/DUCTCRT are raw diagnostics, not this target. Historical optional scalar
  `strut_damage` severity can be NaN while binary fracture is fully known; no
  partial aggregation decision blocks these binary labels. Carry is disabled.
- UT selects BodySet and excludes830 grips. Native struts:2319UT/2259FT.
  Original joints:800UT/788FT. Initially absent FT geometry is not failed.
- Disordered joints derive from selected-mesh degree, unique reference identity
  within the producer's +/-sqrt(2) mm coordinate bound, unchanged boundaries,
  straight FE chains and full canonical-edge equality. Subdivisions vary2-6UT,
  1-6FT; do not infer membership from labels or a fixed element count.
- Audit-canonical reference order is sorted(y,x), **not DATA/DUAL node order**.
  Match specimen provenance and original coordinates, remap endpoints explicitly,
  retain native FT/presence masks and record unmatched/dropped samples. Numeric
  ID397/747 alone does not prove correspondence with an archived ML sample.
- Damage products contain21 actual frames including0. The active ML runner
  drops frame0 by default. Match saved step/frame/time/load metadata; remove0
  only to match the verified consumer configuration. Never ordinal-stack21 with
  20 frames, interpolate binary labels or invent between-frame onset.
- Preserve the existing7137/793/881 specimen partition, seeds42/43/44 and locked
  test exclusion where corresponding data exists. Group related/duplicate files
  by underlying design; never split frames or elements as independent specimens.
  Check a fixture's partition before evaluating any saved model on it.

## Next actions in Improve ML Accuracy

1. Synchronize the receiving checkout and reconcile its current checklist,
   live queue and data availability. Reuse the existing authenticated HPC session;
   leave it open. Confirm exact archive/source identity rather than assuming a
   local path or filename maps to the current training data.
2. Run a **bounded cluster contract/consumer replay on these six fixtures**:
   original identity, endpoint remapping, frame0 alignment, task/absence/unknown
   masks, STATUS-derived parent rule and save/reload. Sixteen focused local tests
   already pass. Standard Python can read NPZs; actual ODB replay requires Abaqus
   Python and matching INPs via the p1 route, not an assumption that the p2 GPU
   environment exposes odbAccess. No new FEA is needed.
3. If useful, prepare post-hoc fracture-region diagnostics for existing saved
   predictions only after exact geometry/source and permitted split membership
   match. This can diagnose displacement errors; it does not establish a trained
   fracture predictor or an improvement from damage supervision.
4. Before supervised damage training, inventory label coverage read-only and
   agree an explicit data/split/extraction plan with the researcher. The few
   selected fixtures cannot establish generalisation. Broader extraction and
   architecture/loss decisions remain explicit scope choices.
5. Once an isolated model test is agreed, retain the selected Case1(i) input
   policy and an unchanged matched control. A symmetric endpoint-embedding strut
   readout is an option, not an existing implementation. Avoid new tokens or
   graph changes in the first auxiliary test; never feed true future damage into
   inference or remove inference edges with it. If field generators change and
   correction won, refit the required TRAIN-only OOF/nested correction banks.
   Select damage weights and train-only balancing explicitly; spatial0.1 and
   existing curve coefficients are not damage-loss optima. Retain the fixed
   field/curve selection and safeguards; add binary precision/recall, prevalence
   and onset diagnostics without substituting them for that comparison.
6. Use the existing preview-first staging, immutable sources, manifest and
   disposable GPU/save/reload preflight before matching production training.
   Preserve the established resource policy, successful-gate dependencies and
   seeds; determine the actual job count from the agreed test rather than this
   brief. Execution success is not accuracy evidence. Keep Windows raw runs in
   Z:/p2 and provide one conclusions-first visual report per study in samples,
   including controls, seed variation, full UT/FT crack-tip views, errors and
   class prevalence. Preserve current preflight retention/failed-run cleanup.

## Evidence and limits

All source hashes stayed unchanged. All audited binary labels are known.
The disordered pass independently matched583422 raw STATUS values, all192276
parent/frame failure/known masks and277820 section SDEG values at five frames/file.
The periodic pass matched296079 STATUS values and140990 section SDEG values.
Sixteen focused tests pass; repository validation reported80 non-failing contract
checks and14 non-failing changed-surface checks, including explicit syntax-only
and skipped coverage. Those are distinct from the real Abaqus checks.
No HPC execution, damage training, learned fracture accuracy or scientific
improvement is claimed by this handoff.

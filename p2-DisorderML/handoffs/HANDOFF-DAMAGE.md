# Damage Modeling -> Improve ML Accuracy: completed FCC dataset handback

Dataset handback19 October; receiving ML workflow reconciled22 October 2026. From **Damage Modeling**
(01a13c9f-c95e-7ff1-854f-a3b2f4fbea6c) to the existing Windows
**Improve ML Accuracy** task (bc645215-a354-451c-93f7-540e7d36d3dc).

## Outcome and ownership

**Damage-label processing, validation, local/HPC delivery and cleanup are complete.**
No required processing action or unresolved label decision remains in Damage
Modeling. The researcher hands actual damage ML implementation, training and
testing back to Improve ML Accuracy. No damage DATA/MODEL loader or prediction
head is implemented yet; no trained damage accuracy or benefit is claimed.

Read applicable instructions, current p2 PROJECT_STATUS and the receiving task's
active workflow before acting. This replaces the older six-fixture-only handoff,
not that task's current experiment plan. Do not repeat extraction, bank building,
transfers or the completed fixture replay. This Markdown submits no jobs and
starts no scheduled follow-up.

## Ready-to-use datasets

Local directory: `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata/`.
HPC directory: `/data/SEMS-TaoLab/Niccolo-Forte/p2/MLdata/`.

| NPZ bank | Ordered samples | Native struts | Stored frames |
| --- | ---: | ---: | ---: |
| UT-UT-disNodes-allD.npz | 8,882 | 2,319 | 21 |
| FT-FT-disNodes-allD.npz | 8,928 | 2,259 | 21 |
| MULTI-UT-disNodes-allD.npz | 8,811 | 2,319 | 21 |
| MULTI-FT-disNodes-allD.npz | 8,811 | 2,259 | 21 |

These additive `strut_status_bank_v1` banks preserve the corresponding
allFIELD IDs/order, existing filtering and periodic ID0. MULTI UT/FT IDs match.
Choose banks by the existing consumer's mech_mode, exact saved IDs and split,
not by whether its networks are independent. Independent Case1(i) models may
still share the paired MULTI population; retain that population when matching
its controls. Use individual UT/FT banks only for the corresponding individual
populations. Do not silently intersect or expand populations.

Each NPZ contains its arrays and metadata and is sufficient for loading.
MLdata and its subfolders contain only files directly consumed by ML algorithms;
optional JSON audit reports are outside MLdata. Both locations retain their
19 NPZ/CSV algorithm inputs. Original FIELDu, curves and simulation data are
preserved; no broad archive/data reorganisation is required.

## Contract the new training consumer must preserve

- `D[sample, frame, native_strut]` is binary fracture: **1 if any constituent
  whole element has STATUS=0; 0 if all constituent statuses are known active;
  otherwise unknown**. Train/evaluate only where `valid_mask` is true.
  Storage0 under a false mask is not an observed negative label.
- SDEG and historical scalar-severity NaNs are diagnostic, not this target.
  UT is body-only; initial FT gaps are absent geometry, not failed struts.
- `edge_index` explicitly maps native strut endpoints into saved task-node
  order. Preserve task geometry, native axes and absence masks. Never broadcast
  strut labels onto nodes, infer membership from consecutive element IDs, or
  substitute the old audit-canonical node order for the ML order.
- Match samples by exact IDs/order and verify their saved geometry/source
  fingerprints. Retain existing TRAIN/validation/locked-test membership and
  related-design grouping. Do not split frames or elements as independent samples.
- All banks store 21 frames including frame0. If the configured ML consumer drops
  frame0, apply the same slice to D, masks and frame metadata. Align by saved
  frame index/order as for displacement fields; do not interpolate binary labels.
- Eleven short UT histories contribute **50 unavailable sample frames and
  115,950 masked strut/frame entries in each UT bank**; FT has none.
  `sample_frame_count` and `frame_present` distinguish observed prefixes
  from unavailable trailing frames; their actual times are NaN. Preserve these
  masks through losses, metrics, batching and save/reload. Never carry failure
  into unobserved frames. Observed unknown STATUS counts are zero.
- Keep true future damage out of inference inputs and topology updates. A
  proposed strut readout/loss is a new training implementation, not part of the
  completed exporter. Its design and comparison belong in the receiving task.

## Merge with the current Accuracy workflow

Current PROJECT_STATUS supersedes historical queue/hold statements:

- Case1(i) uses independent UT/FT models and separate field/curve training.
  The22 October cost-aware decision selects `mixed`:50/50 predicted/true curve
  training and predicted-only deployment, without correction. The original
  corrected_mixed numerical input-study result remains comparison evidence.
- The active loss continuation is `independent-losses-261022-mixed`: gate32868320
  and18 curve jobs32868321–32868338, reusing12 compatible18 October field banks
  from92d844f. Submitted curve source is20474b3. These are submission records,
  not a fresh live-state check. Reconcile PROJECT_STATUS/queue/results; do not
  restart banks, revive cancelled corrected-source jobs or alter frozen runs.
- Put damage in that task's checklist as a separate follow-on with an unchanged
  matched control. Preserve the selected input policy, split 42, seeds 42/43/44,
  fixed field/curve selection and locked-test safeguards. Do not silently fold
  damage into the ongoing loss suite or announce an architecture/loss choice
  that has not been made there.

Receiving work:

1. Implement explicit bank loading/ID/geometry joins and native-strut target
   support in the maintained ML architecture. Validate masks, frame0 handling,
   padding, class counts and save/reload on the actual banks. The old fixture
   replay does not validate a new full-population training consumer.
2. Define the isolated damage prediction/supervision experiment and control,
   including strut readout, loss, train-only class balancing and selection.
   If new field generators are fitted, regenerate their required TRAIN-only OOF
   fields. The current Mixed route has no correction bank to regenerate or reuse.
3. Follow the receiving task's authorised preview/immutable-source/GPU-gate
   workflow before full training. Use current resource policy and its actual
   job schedule. This handoff creates no new job count or automatic monitor.
4. Compare identical permitted validation specimens and masks. Retain existing
   displacement/curve metrics; add positive-class precision/recall, prevalence,
   false positives and onset/spatial diagnostics for damage. Prevent abundant
   intact labels from making plain accuracy look informative. Select on
   validation only; locked-test testing follows the established frozen-policy gate.
5. Produce a Markdown visual report in canonical Z: samples: matched controls,
   seed variability, full UT body and FT crack-tip views, true/predicted fracture,
   errors, masks and limits. Save ML runs under external `Z:/p2/`. Execution
   success and correct labels do not establish learned predictive accuracy.

## Completed processing and evidence

- Array 32085486 completed 91 batches/18,002 records with zero errors. All 18,002
  raw FIELDd NPZ/JSON pairs are locally checksum-verified: 18,000 disordered and
  a separate periodic UT/FT pair. No required source product is unavailable.
- Bank job 32095821 completed 0:0 using padded-builder snapshot 4c435cf; extraction
  used e92e899. Superseded bank 32085487 was cancelled before execution; never
  resume it. Use current maintained code for future work, not archived launchers.
- Recorded corpus checks cover 2,622,439,219 raw STATUS positions and 865,222,188
  parent/frame labels, with zero missing/observed-unknown/reactivation cases.
  Independent native ODB comparison is bounded to **eight products**, 1,171,296
  raw positions and 384,552 parent/frame labels, not the complete corpus.
- The frozen bank builder checked binary arrays, masks, onset and serialization
  round trips. Delivery proof separately checks full NPZ byte identity,
  IDs/order, source fingerprints, geometry/layout and coverage; D/mask content
  inspection there is header-only. Do not enlarge that validation claim.
- Earlier six-fixture CUDA replay 31976809 completed 0:0 on 16 October (51s),
  device=cuda; masks, endpoint/frame layout and reload passed. Saved confirmation
  is in `samples/hpc-cleanup-261017/job-evidence.json` (17 October capture),
  including its DAMAGE_CONTRACT_REPORT log. Older PENDING text in the dated
  `damage-hpc-contract` report is superseded. The replay did not train,
  reopen ODBs, join specimens to predictions or evaluate prediction accuracy.
  Its current temporary-home cleanup is not rechecked here. The receiving task's
  existing researcher-owned home/scratch/archive convention review remains
  separate; neither replay success nor this dataset closeout completes it.

Canonical report/evidence root:
`Z:/00-PhD-gitRepo/p2-DisorderML/samples/damage-dataset/`.
Start with README.md and corpus-summary.json; execution/ holds 91 transfer
receipts, bank-delivery-proof.json, archive-layout.json, hpc-layout.md,
source-equivalence and frame-padding evidence. Full bank hashes are there.

Current completed-job archive, following the researcher's convention:
`/data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data/disNodes/0.2/FCC/damage-261016/`.
It directly contains 94 copied output logs, execution.tar.gz (120 historical
non-log launch files), bank-reports/, manifest.json and README.md, with no extra
provenance directory. Home retains the 94 log originals; 91 production scratch
roots and task-owned staging/helpers are removed. The scheduled follow-up was
**deleted** at the researcher's request; do not recreate it.

Raw exports remain in original HPC job zip/transfer directories. Local copies
are flattened into `Z:/p1/data/Ti/disNodes/0.2/FCC/transfer/` and
`Z:/p1/data/Ti/per/FCC/transfer/`, retaining source-job identity in records.

## Maintained implementation and older fixtures

- p1 SIMscripts/A2_DamageOUTpostProcess.py and A2_DamageBatch.py: STATUS export
  and explicitly bounded FCC batches; B2_ABAQUS-PPscratch-new.sh: damage-only route.
- resources/damage.py: checked INP-chain mapping and missing-aware reduction.
- resources/damage_data.py: aligned banks and staging guard.
- p1 SIMscripts/DAMAGE.md: commands and array specification; p2
  HPC/studies/Damage/review_damage_dataset.py: saved-product/corpus visual report.

The earlier six audit fixtures remain in canonical Z: samples:

| Samples-relative NPZ | Role |
| --- | --- |
| damage-audit/UT-periodic-damage.npz | periodic UT |
| damage-audit/FT-periodic-damage.npz | periodic FT |
| damage-disordered/UT-397-maxD.npz | selected high-ductility UT |
| damage-disordered/UT-747-maxB.npz | Pareto-pair UT |
| damage-disordered/FT-747-maxB.npz | Pareto-pair FT |
| damage-disordered/FT-747-maxK.npz | selected high-toughness FT |

They are diagnostic fixtures, not the training population: the two FT responses
are duplicates, giving five unique task responses across three designs. The
Documents checkout's three generated damage report packs are now absent; use
canonical Z: artifacts, not old paths. The old May test 11233922 contains no
scientific outputs and was left untouched; optional disposal is not a label
processing or ML prerequisite.

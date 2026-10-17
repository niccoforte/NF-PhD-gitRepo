# p2 Implementation Status

This is a compact repository handoff, not a second research plan or an Obsidian mirror. Scientific objectives, manuscript reasoning and supervisor decisions remain in the vault's `Writing/p2/` and `Writing/Research Plans/`. Durable contracts belong in `AGENTS.md`; completed changes belong in Git history.

## Active implementation

- Mac checkout backlog is included in the authorized repository synchronization:
  the pending dual-notebook cells and saved outputs, field-review metadata and
  crack-topology regression assertions are preserved alongside the latest remote
  changes. Final dual checkpoint/result saves retain explicit run-layout metadata,
  protected by the existing fresh-notebook regression. The old `.gitignore`
  staged edit cancels against its working copy;
  `/data/` remains ignored. These saved outputs are historical, not a new training
  or scientific validation run. Future completed repository work is committed
  and pushed under the standing instruction in root `AGENTS.md`.
  Validation: 14 field-loss tests and the synthetic fresh-notebook
  build/train/save/HPO-call regression pass; changed-surface12 and contract85
  checks are non-failing, zero failed. Real data, Abaqus and cluster execution
  are not revalidated by this synchronization.

- Artifact storage consolidation: canonical reports/examples are in
  `Z:/00-PhD-gitRepo/p2-DisorderML/samples/`; saved ML runs stay in external
  `Z:/p2/`. All three Windows checkouts were inspected: none has root `data/`.
  The Documents damage-audit, damage-disordered and damage-dataset packs
  (278 generated files,146,065,480 bytes) are copied to Z: and SHA256-verified.
  Source deletion was blocked by tool policy, so Documents originals still need
  removal; retain its tracked samples/AGENTS.md. Receipt:
  `Z:/p2/diagnostics/storage-cleanup-261017/migration.json`; report index:
  Z: `samples/README.md`. Older Mac data/examples have not yet been reconciled
  from the promised ZIP. Notebook/transfer defaults no longer create repo data.
  Nine follow-up contracts and the fresh Outputs build/train/save/HPO-call check
  pass; changed-surface15 and contract85 checks are non-failing/zero failed.
  Transfer commands were stubbed for destination tests; no archive download or
  native macOS execution was needed for this storage change.

- Existing FCC periodic/20% nodal-disorder STATUS dataset delivery is complete.
  Production array 32085486 finished all 91 batches/18,002 records without errors;
  corrected CPU bank job 32095821 completed 0:0 in 1 h 13 min using bank-code-4c435cf.
  All 18,002 raw FIELDd NPZ/JSON pairs and four allD NPZ banks are checksum-verified
  locally against HPC. Exact existing IDs/order remain UT 8882, FT 8928 and
  MULTI 8811 per task, including periodic0. Both archive locations now contain the
  new banks; existing displacement/curve products and filtering are preserved.
  Recorded checks cover 2,622,439,219 raw STATUS positions and 865,222,188
  parent/frame labels, with zero missing/unknown observed states or reactivation.
  Independent native ODB comparison remains limited to eight preflight products.
  Eleven short UT histories account for 50 absent sample frames/115,950 masked
  parent-frame entries in each UT bank; FT has none. Explicit counts/presence
  preserve raw prefixes and unknown trailing padding without interpolation/carry.
  Human report: samples/damage-dataset/README.md. It derives completion from all
  attempts, existing ML exclusions, 91 transfer receipts and four canonical bank
  proofs, and keeps the eight geometry/evolution examples separate. The delivery
  proof verifies hashes, IDs, source/field layouts and coverage with D/mask header
  inspection; binary/roundtrip validation was performed by the frozen builder.
  Execution records include HPC layout, padding/source-equivalence/replacement
  evidence, source changes and cleanup. All 91 production scratch roots and local
  staging/helpers are removed. Home now retains only 94 successful output logs,
  also copied alongside the completed job's archived results;
  its 120 non-log launch files are checksum-verified in the FCC archive's
  damage-261016/execution.tar.gz. Bank JSON audit reports are in that
  run folder's bank-reports/, outside both MLdata folders. Each MLdata
  folder retains its 19 NPZ/CSV inputs, with unchanged bank hashes. The full
  processing archive is /data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data/disNodes/0.2/FCC/damage-261016.
  The follow-up was deleted at the researcher request and was not recreated.
  Other lattices, new FEA and ML architecture/training integration remain outside
  this completed data-processing task.

- Case1(i) is selected for the next two studies. Input variation is prepared first:
  five source policies (predicted, true, 50/50 mixture, corrected, corrected/true
  mixture), every deployment path using predicted fields. Three nested correction
  banks/four new curve fits per seed give15 full jobs; existing predicted controls
  are reused. Custom-loss raw-P draft comprises30 full jobs: spatial/temporal/both/sudden
  banks and adapted curves, plus combined-curve fits without location and with
  soft/soft peak location per seed. Original
  fixed curve readouts and source substitutions diagnose adaptation separately.
  Both use the existing trainer/runner, matched splits/seeds42/43/44, max450 epochs,
  early-stop52, disposable8h preflights and240h full resource requests. Preview-first
  `HPC/DualOutputs/B4_Independent-followups.sh`. Input suite
  `independent-inputs-261015` submitted from c0ab41a: gate31869463 and15 production
  jobs31869464–31869478. Fresh16 October check: gate COMPLETED0:0; correction-bank
  jobs31869464/31869469/31869474 RUNNING, twelve curve fits dependency-held.
  Full resources verified240h/1GPU/12CPU/90000MB. Home manifest:
  `/data/home/exy053/p2/MULTI/Dual/Transformer/independent-inputs-261015/jobs.tsv`.
  Custom-loss submission ON HOLD: first select the input-policy winner, then
  integrate it into the loss launcher/reference/gate and re-evaluate job counts.
  Do not assume raw P; changed generators require new corrector fits if selected.
  Current coefficients are fixed starting presets, not HPO-selected optima.
  True-trained curves
  also export held-out true-field evaluations; the gate now checks that artifact.
  **HIGH PRIORITY manual task:** researcher personally reviews HPC home, scratch
  and archive ordering/names/conventions; recorded under Cleanup in the active
  accuracy checklist. This manual review is not marked completed by agent checks.
  Eight CPU follow-up tests pass, including platform-specific notebook paths,
  true-trained/true-field export agreement, complete nested-source fits,
  frozen-generator equality, saved correction reload, loss serialization and
  mocked15/30-job dependencies. Eight prior independence,14 field-loss and10 sudden
  tests pass. Current path/evaluation change: changed-surface13 and contract80
  checks non-failing/zero failed.
  The broader31-test dual suite has one skipped
  test and one Windows SQLite temporary-file cleanup error in the unchanged HPO
  test (its training/resume assertions pass); no HPO implementation changed.
  Windows raw runs have been relocated by the researcher to `Z:/p2`; the three
  dual/HPO notebook configurations and operational guidance now use that root.
  The fresh Outputs build/train/save/HPO-call regression passes. General visual
  report guidance is in root AGENTS/README; existing reports stay in ignored samples.

- Independence review: gate31448215 and all30 production jobs31448216–31448245
  COMPLETED0:0. All24 case fits/six banks downloaded to ignored
  `Z:/p2/MULTI/Dual/Transformer/` after researcher relocation;591 JSON/NPZ/checkpoint/history
  files match remote SHA-256. Human report/figures: `samples/independence-review/`.
  All1356 files match remote sizes. All24 prediction arrays pass ID, train-scaler,
  mask/truth identity and independently recomputed physical-RMSE checks; frozen
  fields match exactly between their two consumer cases. Eighteen figures include
  seed variation, safeguards, diversity, full-body UT/crack-tip FT truth/prediction/
  error maps, curves with error bars and log loss histories. Guidance/changed-surface
  validation passes (six non-failing/zero failed); no training implementation changed.
  Case1(i), four independently trained models, is the practical recommendation
  under this fixed configuration. Three-seed mean validation RMSE skill (%) is
  UTfield6.11, FTfield24.20, UTcurve4.55, FTcurve18.73; joint3(ii) gives5.09,
  13.30,3.43,−0.17. End-to-end FT curves collapse across all seeds, also in1(ii)
  without any UT/FT sharing. Shared fields worsen FT; shared curves on identical
  frozen fields do not improve mean accuracy. This is not proof of gradient
  interference or globally optimal architecture. Keep locked test untouched and
  agreed2% field margin; curve/local acceptance tolerances remain unagreed.
  The user has selected Case1(i); follow-up preparations are described above.
  No legacy defaults or historical jobs/checkpoints were changed by this selection.

- Handoff freshness review: optimisation now records the completed property-oracle
  evidence and architecture-neutral boundary while sharing selection is pending;
  physics/strain distinguishes implemented CPU-tested sudden weights from unproven
  accuracy benefit. Damage and Tokenisation need no additional scope or changes.

- Damage handback is integrated into the current Accuracy checklist. Bounded
  cluster fixture/consumer checks are separate from the newly authorised FCC
  dataset export above; no damage-training comparison is defined. Six ignored audit
  NPZs are now in the Z: checkout's samples, not supplied by Git; exact
  hash-verified transfer precedes cluster execution.
  New `HPC/DualOutputs/A0-HPC-Damage-contract.py` packages only these six NPZ/INP
  pairs and a saved Case1(i) DATA descriptor, then replays masks/reference endpoints,
  frame0 removal and NumPy/Torch/device reload. Four new and16 producer tests pass;
  six real CPU layout replays passed, no ODB reopened. All800 reference indices
  remap; disordered actual/shared times differ by at most2.7567e-7s. No time
  interpolation, specimen join or prediction evaluation follows from this result.
  Local pinned bundle: `Z:/p2/diagnostics/damage-contract-261016/fixtures/`.
  Disposable B1 GPU check31976809 submitted from aa1d2e0,1h/1GPU/12CPU/90000MB;
  current PENDING/Resources, no GPU result yet. All13 remote file hashes match.
  Home check: `/data/home/exy053/p2/MULTI/Dual/Transformer/damage-contract-261016/`.
  After success retrieve the home-log JSON and remove only its unused temporary
  fixture/source copies; no preflight archives. Human report/two figures:
  `samples/damage-hpc-contract/README.md`. Changed-surface9 and contract82 checks
  non-failing/zero failed; these do not establish GPU/ODB execution.

- The researcher requests damage handback to the current Windows **Improve ML
  Accuracy** chat for HPC tests. `handoffs/HANDOFF-DAMAGE.md` is the reconciled
  source-attributed transfer. Preserve input-study priority and the custom-loss
  hold until its input-policy winner is selected. Bounded cluster contract/data
  alignment checks can proceed first; any damage-training ablation requires a
  sufficient explicit label corpus and a separately agreed model/loss plan.
  No damage DATA/MODEL loader/head exists; the six selected fixtures include
  duplicate FT evidence. The recipient owns the existing checklist and fresh
  queue reconciliation. This handback submits no jobs or claims HPC results.

- Damage Modeling: periodic UT/FT ODBs read in Abaqus 2024 Python; the researcher
  additionally authorised exactly four local disordered files: UT 397 maxD,
  UT 747 maxB, FT 747 maxB and FT 747 maxK in the same FCC sims directory.
  Separate exporter `SIMscripts/A2_DamageOUTpostProcess.py` under p1 composes
  `resources/damage.py`; p2 `code/review_damage.py` reviews versioned saved NPZs.
  Original mesh IDs/connectivity/coordinates match; source SHA-256 hashes are
  unchanged. Periodic UT BodySet: 2,319 struts/10,075 elements (4/5 per strut),
  excluding 830 grip elements. Periodic FT: 2,259/4,024 (1/2/4/5). Both initial
  endpoint sets exactly match `reference_field_edges`. Each has 21 recorded
  frames, zero missing expected
  SDEG/STATUS records, section points 1/5 (bottom/top), and no STATUS reactivation.
  Final verified failed parents UT 83, FT 7 (one failed element per affected parent).
  Stored SDEG reaches only 0.99999; whole-element STATUS=0 establishes failure.
  The researcher clarified the target as binary fracture from STATUS: use existing
  `strut_failed`/`strut_failure_known`, with 0 for known all-active parents and 1
  for any failed member. Raw SDEG is optional diagnostic context; no partial
  reduction choice blocks this target. Historical scalar-severity NaNs remain
  separate from the known binary labels. The periodic audit passed twelve focused
  tests and 80 contract checks, zero failures; all 96,138 binary parent/frame
  labels independently match raw STATUS, including fully known states where
  severity is NaN.
  Periodic report/raw examples/figures/NPZs: `samples/damage-audit/`; the separate
  disordered comparison belongs in `samples/damage-disordered/`.
  Those audits changed no architecture or source data. The subsequent authorised
  FCC bulk pass is recorded above; model integration remains separate. User clarified D=highest UT ductility,
  K=initial FT toughness, B=same-disorder UT/FT property-Pareto pair.
  Periodic independent live-ODB checks matched all 296,079
  STATUS records/all-frame parent failures and 140,990 section SDEG spot checks.
  Implementation/guidance are tracked; generated research artifacts remain
  ignored under the existing samples policy. Report leads with conclusions
  and binary fracture figures: purple0/red1, black-dashed unknown, no grey active
  state. Of 109 UT parents observed active with surface damage, 33 later fracture
  and 76 remain active at the final frame. Example2253/e535 never fails through
  frame20, even after both reported outer section points degrade. Interior points
  are not saved. Parent-fate CSVs preserve this observed-window evidence. The
  tracked runbook
  starts with usage commands; the full specification is optional reference.
  Disordered INP provenance validates graph-degree original joints, unique
  reference identities within producer-derived +/-sqrt(2) mm coordinate bounds,
  straight FE chains and exact canonical edge sets. UT 397/747 have 9,463/9,461
  selected elements, 2,319 parents each and 830 excluded grips each; FT has
  4,429 elements/2,259 parents. All 788 shared maxB joints match exactly across
  UT/FT. All four ODB exports have 21 actual frames, unchanged source hashes,
  zero missing expected SDEG/STATUS values and no reactivation. Final failed
  parents/elements are UT 397 95/97, UT 747 92/93, FT maxB 9/9 and FT maxK 9/9.
  Every binary state is known across 192,276 parent/frame labels. FT maxB/maxK
  INPs are byte-identical and all exported STATUS/SDEG, displacement, loading,
  reaction and raw history arrays match exactly; these are duplicate exported
  responses, not independent specimens despite different ODB hashes. Sixteen
  synthetic mapping/reduction tests pass. Independent live-ODB verification
  matches all 583,422 STATUS values, all 192,276 parent/frame failure and known
  masks, and 277,820 section SDEG values at five frames/file. The separate visual
  report retains all four file results and their provenance.

- 14 October retention cleanup: all56 preflight archive directories from nine
  COMPLETED0:0 jobs removed permanently; successful home logs retained. Failed
  original independence scratch/archive/home logs were already removed (54 paths).
  Replacement gate31448215 passed; all full jobs subsequently completed, not deleted
  or resubmitted. B1 now skips preflight archiving and removes successful scratch;
  failed evidence cleanup after correction is a high-priority HPC instruction.

- Recovery13 October: original gate31417279 COMPLETED0:0; all18 started full jobs
  FAILED1:0 before training at the string-versus-integer split-ID comparison.
  Twelve dependent staged jobs never started; queue empty before cleanup.
  Normalized both sides without relaxing membership/order. Regression added;
  replacement GPU gate includes one full-data epoch before the subset suite.
  User-authorized cleanup removed18 failed scratch directories,18 metadata-only
  archives and18 home logs (54 paths); no checkpoints/predictions existed there.
  Raw failed files are permanently removed; successful preflight, source/manifest
  and concise failure evidence remain. Replacement independence-261013-r1 submitted
  from6b0db82: gate31448215; six banks/24 cases31448216–31448245 dependency-gated.
  Full requests remain240h/1GPU/12CPU/90000MB. Eight focused tests passed;
  base-ref changed validation10 non-failing/zero failed. GPU gate subsequently passed.

- Original `independence-261013` (sourceec9189d, gate31417279/full31417280–31417309)
  is superseded by the completed replacement above. Its home source/manifest retain
  provenance; failed useless outputs were cleaned after the guard correction.
- Verification for the independence preparation: 48 dual regression tests passed (including seven isolation/OOF/checkpoint/launcher tests); changed-surface22 and contract76 non-failing/zero failed. Existing notebook, `.gitignore` and field-test edits remain untouched. Names1(i),1(ii),2a(i),2a(ii),2b(i),2b(ii),3(i),3(ii) use predicted-field deployment, seeds42/43/44, matched configuration and equal normalized MSE. Staged cases reuse three-fold training-only OOF banks with inner selection and fold-only normalization. High-priority AFTER sharing selection: matched custom-loss comparisons and predicted-only/true-only/mixed curve training, ALL evaluated on predicted fields. True-only training removes the curve-to-field gradient. Gradient logging, other features/weights and correction remain deferred. The2% field margin is accepted; numerical curve/local safeguards remain to agree before final selection.
- Retirement audit complete13 October: read full available user/assistant histories of Assess unified ML model feasibility, Summarize GNN chat changes and Review cleanup folder from the reconnected PC. No extra experiments added: useful missing ideas were already retained in the accuracy brief or CurveOutputs cleanup notes; obsolete code/revival proposals stay superseded. These three chats may be archived. Keep the active accuracy checklist; no chats were automatically archived.
- Source reconciliation complete: the full available user/answer history of **Dual Model** (`01a0a5ba-1e2c-7682-92c7-cefabb96a6e8`) and the accuracy brief have been reconciled into **Improve ML Accuracy**, against current code and the user's newer decisions. Dual Model may be archived; this is transfer of ownership, not completion of every experiment. Keep `handoffs/HANDOFF-ML-ACCURACY.md` for now: the run guide/index still link to its detailed notebook backlog, controlled-gallery/pin-tolerance checks and per-test visual review requirements. Also retain the proposed task-embedding usage diagnostic (zero/swap at fixed weights, with gradient/update checks); it is not a completed scientific result. Retire the brief only after these details have a maintained home and all incoming links are updated; explicitly tell the user then. No completed runs, cancelled proposals or sibling-task work were reactivated, and no chat/file was deleted or archived in this audit.
- Current review decisions: all seven continuation/index files are consolidated under `handoffs/`; implementations and notebooks stay in place. No definitive four-output winner or default promotion: true-field auxiliary supervision leads the ten matched joint runs; crack-face is the only numerical four-output improvement, with a tiny UT-field gain. Existing independent spatial-loss results remain promising but are not matched to the dual population. User parks further thickness audits; retain provenance without adding thickness features or claiming no effect. Remove thickness from presentation scope; do not change a deck while its working revision/ownership is unresolved.
- Sudden/non-affine weighting remains opt-in, CPU-tested and not an established accuracy improvement. It is excluded from the submitted independence comparison above. Real specimen8701 illustrations are in `samples/accuracy-continuation/dynamic-weights-8701/`:40 frame maps, UT/FT overviews and provenance. Both INP geometries and saved train-only scales/edges were checked; figures show retrospective target-derived weights only, not training outcomes. Grouped loss-history plots distinguish independent fits while retaining log scales. Existing dirty notebooks/.gitignore/field-test edits remain untouched.
- Validation for this preparation: 31 existing DUAL +10 new sudden/reporting +14 field-loss tests passed; changed-surface22 and contract74 checks non-failing, zero failed. Guidance passes with existing instruction-chain size warnings. Refreshed local comparison reports lead with relative errors; source-comparison figure visually checked. Human worked weighting example: samples/accuracy-continuation/sudden-weighting.md. Scientific/GPU behaviour remains unverified; old archived metrics and embedded notebook outputs were not rewritten.
- Presentation continuation is an in-chat handoff for Improve ML Accuracy (2), not a new file or deck edit. Read-only inspection found `progress.pptx` still at 212 slides and `progress-supervision-20261006.pptx` at 207 (changed since its 208-slide delivery). Reconcile the intended working deck before future edits; use titles/stable slide IDs, not historical numbering. The revised deck already has sharing, source-mismatch and property-method slides; update results without duplication. AcademicWriting local presentation guidance is newer than the shared ancestor; ff-only pull stopped on unrelated remote LaTeX divergence, with no merge/reset.
- Latest read-only SSH attempt on 8 October failed authentication; peak jobs30813256–30813260 cannot yet be rechecked. No new submissions or duplicate runs. QMUL's login documentation explicitly requires an account password in addition to a protected SSH key; an agent-loaded key alone does not establish passwordless access. Reuse a user-authenticated ControlMaster, never bypass policy or request secrets in chat. HPC documentation sync remains blocked; preserve immutable job snapshots.

- 8 October: renewed SSH verified property preflight/full 30367256/30367272 and residual/true-field auxiliary 30326584/30326586 COMPLETED 0:0. All four archives downloaded through B3 and checksum-verified, including checkpoints/predictions/metrics/logs. Ten joint runs share exact 7137/793/881 split IDs. New scores: residual .837071, true_field .796952 (lowest joint score, but FT field RMSE worsens ~1.2%). Residual improves FT field ~4.7% but worsens UT field ~5.9%. Single-seed validation, no new default winner. Full property oracle: 85 epochs, best33, 793 validation IDs; RMSE reductions versus training mean: Strength34.16%, Ductility51.09%, WoF80.64%, K_JIC73.49%. Saved truths/metrics independently checked; no predicted-field property or locked-test evaluation. Reports/figures: samples/accuracy-continuation/README.md and field-to-property.md. Existing dirty notebooks/.gitignore/field-test edits preserved.
- Corrected peak-loss suite submitted from immutable c29d77f after clean HPC fast-forward. Gate30813256 (4h); true source without/with location30813257/30813258; predicted source without/with30813259/30813260. Full jobs240h, oneGPU/12CPU/90000MB, seed42/split42, max450 epochs/patience52, all afterok gate and kill-on-invalid-dep. Existing matched MSE controls retained; anchor hashes/configurations/splits verified. At last check gate PENDING Resources; full jobs PENDING Dependency. Home manifest /data/home/exy053/p2/MULTI/Dual/Transformer/dual-peak-261008/jobs.tsv. B1 home→scratch→archive and saved metrics unchanged; no GPU-success claim yet. Prior31 local dual tests and mocked5/13-case dispatch checks passed. Temporary transport bundle removed after verified deployment; shared SSH master left open.
- Thickness/property follow-up: 29 verified specimens (26 train,3 validation), 11 batches, excluding four audited locked-test IDs. Pearson correlations: strength−.074, ductility+.132, WoF+.154, K_JIC+.147; FT Spearman−.096. Earlier FT mean K_JIC ~4.2% higher, only6 specimens with large scatter. Batch-bootstrap intervals broad; no clear simple association, no causal/no-effect claim. Later within-batch thickness constant, so batch/thickness effects inseparable. Archived assignment supports first-UT thickness reuse, not a proven corrected density formula. No feature/filter/data/simulation change. Evidence: samples/accuracy-continuation/thickness-properties.md/.json/.png; original provenance limitations retained.
- Review validation: sample report generator completed with exact split/ID assertions and independent property RMSE checks; all three main scientific figures visually reviewed. Changed-surface validator: 11 non-failing, zero failed. Four B3 checksum dry-runs showed no differing files (only directory timestamps). Bundled analysis runtime lacked SciPy, so the established nf-phd scientific environment was used without installing anything. No new notebooks or slide edits; reports remain in the requested samples directory.
- Final SSH transport dropped after successful downloads/submissions (Broken pipe, authentication failure). Both Git hosts contain the results documentation; HPC checkout and immutable queued-job source are verified at c29d77f. Only documentation fast-forward and a fresh queue check remain blocked. Reconnection requested; do not resubmit jobs or change their snapshots. No master shutdown or SSH configuration change was performed.
- Current handoff reconciliation now explicitly authorised: Tokenisation notes include the observed graph trade-off and response-informed motion motifs; Damage and Surrogate Optimisation briefs retain edge-label and field-objective safeguards. New HANDOFF-PHYSICS-STRAIN.md scopes a separate iterative true-field strain/coupling feasibility task, not an implemented PINN or dispatched chat. Sudden-motion weighting should combine temporal change-of-increment and local non-affine departure; neither large displacement nor a fixed crack region is the intended rule. No new weighting/strain/damage/feature implementation in this turn.

- Accuracy continuation reconciled the current code with latest annotations, without loading other handoff briefs. B3 downloaded eleven completed matched-suite archives: six original runs plus recovery detach/winner_probe/curve_predicted/curve_true/ft_region. At 18:25 BST on 4 October residual30326584/true_field30326586 remained RUNNING; no duplicates submitted. Eight completed joint runs share exact 7137/793/881 IDs. Scores: baseline .863022, crack_face .807569, local_graph .902120, partial .841817, private .827576, late_frame .865194, detach .866487, ft_region .809125. Single-seed validation only; ft_region is a fixed geometric prior, NOT dynamic sudden-motion weighting. Fresh true-field curve fit reaches RMSE UT1.639/FT3039.064 on true fields versus predicted-source fit UT4.360/FT6678.813 on predicted fields; substitution degrades either. Two-curve scores are not comparable to four-output selection. Human plots/evidence: samples/accuracy-continuation/README.md. Existing notebooks/user edits preserved.
- Explicit soft/soft peak loss implemented with saved mode and historical hard-target fallback; hard peak reporting and peak-height terms unchanged. No existing experiment/HPO changed; the new corrected-loss submission is recorded above. Automatic activity weights remain distinct from sudden/non-affine anomaly weights and excluded from the matched suite. Tokenisation plan reconciled with actual correlation-PCA/KMeans prototype; other handoffs retained untouched. samples/accuracy-continuation/test-guide.md covers all 31 current dual checks with expected/observed assertions, tolerances and limits; measured mask-leakage/gradient counterexamples and affine illustrations accompany it. All 31 dual and 14 field tests pass; changed-surface validator 17 non-failing/zero failed. First validator attempt timed out importing under concurrent load; thread-limited retry passed. Notebook comparison integration remains pending; existing dirty notebooks preserved.
- Direct property oracle test submitted through B1 after user confirmed UT Strength/Ductility/WoF and FT K_JIC. Runner: HPC/FieldToCurve/A0-HPC_FieldToProperty-test.py; immutable source47fabb9, published to both Git hosts and deployed to clean HPC checkout. Preflight30367256 (64 pairs/one epoch/4h) and full30367272 (450 maximum epochs/early-stop52/240h) both PENDING at18:25 BST; full requires afterok:30367256 and kill-on-invalid-dep. Both request andrena/pilot_andrena, 1GPU/12CPU/90000MB. Submission is NOT GPU-success or accuracy evidence. Home snapshot: /data/home/exy053/p2/MULTI/FieldToProperty/Transformer/field-property-261004/source; archive root /data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/FieldToProperty/Transformer, labels field-property-preflight-261004 and field-property-261004. Reuses dual encoder plus legacy trainer, true fields only, no disorder/context channels, exact frozen split-ID check, training-only normalization, validation-only scoring. Every frozen ID has an archived property row. Preserve separate FieldToProperty artifact identity; no general curve-loader support claimed. Do not submit duplicate jobs. Verified temporary Git transport bundle removed after deployment; snapshot is required by queued jobs and must remain.
- Authorised bounded thickness follow-up completed: 33 ML-included specimen pairs/66 INPs across 11 batches. Every input CSV coordinate matches exactly; all UT800/FT788 initial field coordinates match INP labels within 5e-7 mm, with matching stems/source basenames. Early sampled batches6538720/6538745 have varying per-task thickness (FT~.412 versus UT~.402); all nine later sampled batches have identical UT/FT thickness across the three inspected specimens in each batch, with small between-batch variation. Sampled1000/1001 contrast agrees with archived unconditional versus if-None assignment, consistent with first-UT thickness reuse. Caveats: batch6892743 B1 says initial1901 despite INPs1001–1100; merged Windows source paths cannot prove exact ODB provenance. No claim of dataset-wide change point, actual density or error effect. No data/model/feature/job changes. Human report and hashed row evidence: samples/accuracy-continuation/thickness-provenance.md/.json; next analysis requires separate scope. Original home bundle/p2-code-transfer check was clean.

- Selective recovery verified 4 October after SSH reconnection: `dual-recovery-261004`, immutable source `0b9215e`. Preflight 30326583 COMPLETED 0:0 in 9m05s on sbg23, all thirteen checkpoint/prediction/metrics/log archives verified; winner_probe 30326588 COMPLETED 0:0 in 2m43s with field-source diagnostics archived. At approximately 15:47 BST residual 30326584, ft_region 30326585, true_field 30326586, detach 30326587, curve_predicted 30326589 and curve_true 30326590 were RUNNING with advancing epochs. Full requests remain 240 hours, one GPU, 12 CPUs, 90,000 MB, andrena/pilot_andrena. Six successful original runs were not repeated. sbg10 is administratively drained for modules; no claim of its repair, no model/loss changes or environment installation needed. Local 29 dual-contract tests and changed-surface/guidance validation passed. README owns IDs, paths and evidence; inspect existing jobs, never resubmit blindly. Accuracy brief preserves controlled-gallery, feature-bundle restraint and scale-aware pin-tolerance requirements. Keep SSH master running; broken pipe cause remains unproven, no SSH config changes made.

- Handoff routing: handoffs/HANDOFFS.md indexes accuracy, damage, context, surrogate optimisation and physics/strain briefs, plus the Tokenisation plan in the same handoffs/ directory. Each reads its own brief, not all briefs or the source transcript. No chats dispatched; data/samples ignore policy unchanged. Storage cleanup is completed here, removed from delegated tasks.

- Original suite status verified 4 October: dual-compare-260930-r1 (source 6a3aee1) has six COMPLETED full runs: baseline 29601763, partial 29601764, private 29601765, crack_face 29601766, local_graph 29601767, late_frame 29601769. Checkpoint/prediction/metrics/log archives verified. Residual 29601768, ft_region 29601770, true_field 29601771, detach 29601772 and winner_probe 29601773 FAILED before Python on sbg10 because miniforge was unavailable; dependent curve fits 29601774/29601775 CANCELLED. No original-suite jobs remain queued/running. Preflight 29601762 previously completed all thirteen modes; this did not guarantee module availability on every node. No accuracy comparison made. Replacement jobs are recorded above; never rerun the six successful comparisons during recovery.

- Storage audit/cleanup completed 4 October with explicit user approval: removed the five empty failed scratch directories above and their home logs, plus original failure log 29598744 (the user had already deleted its empty scratch). Original failure was missing module command; later five were missing miniforge. Logs' archive messages accompanied zero-byte transfers; no failed-run results existed at their conventional archive destinations. No archive deletion needed. All scripts/source snapshots/manifests and successful results preserved; .scratch-expiring-history left untouched. Raw failure logs permanently deleted, no backup; concise evidence in HPC/DualOutputs/README.md. No blanket permission to erase useful interrupted checkpoints.

- The authorised suite covers baseline, partial/private field sharing, crack-face, local graph, residual fields, auxiliary true-field supervision, detached gradients, late-frame and FT-region weights, frozen-winner substitution and two fresh frozen-source curve fits. It excludes component weights, old activity-localisation, damage extraction/training, corrected soft-peak loss, design search and locked-test development. HANDOFF-ML-ACCURACY.md records each job, interpretation and notebook review steps; HANDOFFS.md routes task ownership; do not resubmit duplicates.
- New loss controls: positive raw 1+relative-load weights, and a separate FT raw2-inside/raw1-outside prior using A1's verified reference box (125 retained nodes); valid-mean-one normalization and unweighted selection. Frozen-source curve fits freeze the winner field stage/dropout, train fresh identically initialized curve stages and select on two curve scores. Both source evaluations are saved. Winner has three field blocks and four curve blocks. Human examples: samples/dual-weighting-experiments.md. Local affine-departure weighting is a proposal, not silently implemented. The accuracy brief contains continuation/reading anchors; HANDOFF-DAMAGE.md owns the separate strut-damage task. Existing dirty notebooks/.gitignore/field-loss tests remain preserved.
- Local verification: all 28 dual-contract tests passed, including six runner modes on synthetic 800-node data and the explicit curve-source plotting regression; the prior 14 field-loss tests also passed. Latest changed-working-tree validation: 14 non-failing/zero failed; committed-change validation since 37350e9: 12 non-failing/zero failed, including B1 shell syntax and NumPy CSV/NPZ round trips. Guidance checker passed with size warnings for the accumulated HPC/code instruction chains. Non-interactive module/Conda bootstrap passed remotely. Scientific accuracy still requires the full suite; preflight is execution evidence only.
- Final HPO verified remotely: job 26375307 COMPLETED (0:0), 9d14h05m; the 230-hour Python budget ended the study run with 45 complete/10 pruned/one waiting retry, not 200 evaluations. Winner trial 46 (epoch 58), fixed score 0.82661694 vs earlier trial 37's 0.83263056. Validation RMSE skill vs mean: UT field +4.20%, FT field +16.55%, UT curve +4.06%, FT curve +12.14%; improvement is not uniform. Raw archive/staged code remain unchanged. B3 downloads completed for the full HPO (193 files, 56 checkpoints), twelve loss trials, three preflights and four older field-to-curve tests. SQLite quick_check and actual winner strict checkpoint/review loading passed. Local inventory/report: samples/hpc-archive-review.md; study: data/MULTI/Dual/Transformer/HPO/dual-joint-hpo1 relative to repo root. Older missing HPO branches are listed in that report, not silently downloaded.
- Earlier architecture validation: 23 dual tests, 14 field-loss tests and nine full-800-node synthetic train/save/reload modes passed. Worked arithmetic is in samples/dual-experiment-examples.md. These checks support the submitted comparisons but are not real-data accuracy evidence.
- Git deployment: both hosting destinations and the clean HPC checkout contain the submitted implementation. Later review/documentation fixes do not modify the immutable queued-job snapshot. Preserve pre-existing .gitignore, field-test and notebook changes locally.

- Crack-context review: nominal tip offsets already correctly use the point (120,95) on the first intact vertical strut, not a node. An opt-in `crack_face=True` DUAL_DATA factory option now appends `initial_crack_face` from initial UT/FT connection loss (26 retained FT nodes; UT=0), preserving the default 11-channel schema and existing runs. Sample coordination table corrected: (100,100) FT degree is 5, not 8; strut-length statistics remain explanatory, not inputs. Validation: 17 dual-contract tests, 14 field-loss tests and all 14 changed-surface checks passed locally; no research-data training performed. Broader topology-aware attention/local graph work is handed off to Dual Model; controlled-gallery presentation and feature/loss comparisons belong to Improve ML Accuracy. Neither handoff has been dispatched and no jobs/checkpoints were modified.
- Graph audit resolved with opt-in `MODEL(graph_semantics="fcc_initial_v1")`: validated initial FT cut and bidirectional message edges, historical semantics retained for old/default checkpoints. Counts/indexing pass for native/padded views; corrected legacy GNN training explicitly requires native FT because its pooling is unmasked. Existing pin-circle comparisons can flip exactly-on-circle reference nodes under extreme unit rescaling/translation due to floating-point rounding; define/test a unit-aware boundary tolerance before changing validated pin membership. New crack-context rescaling tests pass independently of those pin columns.

- Current field-loss work: opt-in `resources/MLfield.py` implements signed spatial and temporal differences after physical affine reconstruction, train-only jump scales, degree-balanced spatial averaging and bounded specimen-specific target weights. `FieldOutputs/A0-HPC_Field-lossTrial.py` and `DualOutputs/A0-HPC-Dual-lossTrial.py` prepare fixed-architecture baseline/spatial/temporal/both trials. Existing run scripts and default losses remain intact. Local implementation is complete. On 17 September all three GPU preflights completed successfully (exit 0:0): UT 27230122 in 3m59s, FT 27230123 in 4m51s, DUAL 27230124 in 1m49s. Diagnostics and archives were verified, including frozen-curve intersections of 799 UT / 804 FT validation specimens. Twelve full trials were released by their successful preflight dependencies; seven were RUNNING and five waiting for QOSMaxGRESPerUser at 10:29 BST. No accuracy improvement is established yet.
- Local validation: 14 field-loss and 16 dual contract tests passed; changed-surface validation reported 29 non-failing checks and contract validation 70, with zero failures. Checks cover gradients, masks, degree averaging, affine reconstruction, graph node batches, saved/reloaded losses, frozen curve sample-ID alignment and notebook/runner integration. Scientific accuracy and the full independent real-data curve bridge still require HPC. Samples now include `field-loss-review/implementation-notes.md` (worked arithmetic and deferred decisions), `README.md` (mesh/frame/peak audit), and `field-errors.md` (regenerated labelled validation maps). New motion plots include explicit event counts and qualified timing/orientation; the existing interactive viewers remain intact.
- PowerPoint technical revision on 16 September: fourteen slides, 143–156, replace only the eight prior additions at the end of Machine Learning in OneDrive `Research/progress.pptx`. They include 19 native editable equations, worked spatial/temporal-loss diagrams, adaptive weights, shared-gradient conflicts, strain, damage and local graph/CNN proposals. Proposed methods remain explicitly distinct from implemented losses. All 184 original slide, notes, layout/master and media parts are byte-preserved. New-slide renders were inspected and the 198-slide merged package passed integrity checks. External recovery copies and build/QA records: `/tmp/p2-field-slides-qBcDhL/technical/`. Native PowerPoint application/accessibility checks were not run.
- Bounded archive audit: specimen 8701's two original INPs agree exactly with periodic reconstruction (UT 2,319 struts; FT 2,259). Mesh subdivision counts vary; never assume five/ten elements per strut. All 100 UT + 100 FT raw NPZ exports inspected in archive 10605572 follow nominal 21-frame sampling to within 3e-7 step-time units. This is representative, not an exhaustive dataset audit. Evidence and labelled plots: `samples/field-loss-review/`.
- Historical curve loss has nonzero exact-target error (median 4.88% UT / 3.19% FT of loading span at beta20). Both retrieved full-curve HPO winners used MSE, not CombinedCurveLoss. The explicit soft/soft correction is now implemented as recorded above; the three-way loss comparison remains pending. Do not reinterpret archived losses.
- Publication and HPC: implementation `f680c6d` and handoff `cc06f53` were verified in both hosting destinations and the clean HPC checkout before submission. Thirty focused tests passed again on 17 September; previous full-size synthetic forward/backward checks used the exact UT/FT HPO architectures (800/788 nodes). Unrelated `.gitignore` staging, notebook outputs/counts and earlier user edits remain preserved. The technical-slide task is completed/idle; retain its fourteen-slide revision. HPO 26375307 remained RUNNING after 6d17h at the latest check; its staged source and study were not modified. SSH availability remains temporary.
- Submitted loss trials (17 September): preflights UT **27230122**, FT **27230123**, DUAL **27230124** request two hours. Full jobs, ordered baseline/spatial/temporal/both, are UT **27230164–27230167**, FT **27230168–27230171**, DUAL **27230172–27230175**, each 240 hours, one GPU, 12 CPUs and 90,000 MB RAM on andrena/pilot_andrena. Each has `afterok` on its own preflight and `--kill-on-invalid-dep=yes`. Full labels are `loss-{UT|FT|DUAL}-{variant}-260917`; independent archives are `{UT|FT}/Field/Transformer/<label>`, dual archives `MULTI/Dual/Transformer/<label>`, under `/data/SEMS-TaoLab/Niccolo-Forte/p2`. Home launch/log directories mirror those task/output/model paths under `/data/home/exy053/p2`; scratch is `/gpfs/scratch/exy053/<job-id>`. Do not resubmit duplicates. Independent preflights use full data for the held-out frozen curve intersection, one epoch, both loss terms; dual preflight uses 64 pairs, one epoch and no range split. Full runs use full data and up to 450 epochs with early stopping.
- Latest archive check: all twelve full trials **27230164–27230175** completed with exit **0:0**, as did the three one-epoch preflights **27230122–27230124**. Every full-run archive listed above contains checkpoints, predictions, loss histories, motion diagnostics and curve comparisons. UT elapsed times were 1h11m–1h30m, FT 3h10m–3h35m and DUAL 3h14m–4h28m. No comparative accuracy assessment or download was performed in this archive-location check. Next: compare paired validation results across four loss variants before selecting localisation-weighted follow-ups. No damage/strain extraction, architectural refinement or new HPO yet; motion timing/orientation remain kinematic proxies, not confirmed fracture labels.
- Completed transfer cleanup: removed the three home-level `p2-hpo-*.bundle` files and `p2-code-transfer.pUR8kW` containing `field-loss.bundle` and `handoff-260917.bundle`, after verifying every bundled tip was already an ancestor of the clean HPC checkout. These were Git transport packages, not run data or dependencies. Code/history remain in Git; no datasets, checkpoints, archives, logs or job scripts were removed. Dispose of future temporary transport bundles after verified synchronization rather than accumulating them in home.

- Opt-in joint two-stage UT/FT model: `resources/MLdual.py`. Each stage uses one shared encoder call with task streams stacked along the batch dimension. Field and curve losses train both stages through one optimizer.
- Legacy `DATA`, `MODEL`, separate-task networks and notebooks retain their existing paths.
- Real-data runner and HPO-informed trial preset: `HPC/DualOutputs/`. The adjacent README owns run commands, exact HPO sources and the configuration rationale.
- Dual notebooks: `code/ML-DualOutputs.ipynb` directly constructs DAT/TR_DUAL or loads the saved checkpoint/results, with explicit fresh HPO and train/save/predict cells. `code/ML-DualPostProcessing.ipynb` displays both UT/FT field and curve branches, two live field viewers, log losses and the four-pair 2×2 error plot. `code/ML-HPOpostProcess.ipynb` is the shared single-task/cross-model/dual study review surface; no separate dual HPO notebook. Trial 1 remains the saved-run default; worked inputs belong in `samples/`.
- Human review material: `samples/README.md`, including controlled examples and three actual HPC input reports. Shared reference features and task-specific body-interface/pin/crack features are active. Degree and graph-relative attention remain explanatory examples, not active inputs.
- Existing single-task/HPO notebooks and `HPC/FieldOutputs/`, `HPC/FieldToCurve/` remain the independent baseline surfaces. `handoffs/TOKENIZATION_NEXT_STEPS.md` owns the separate exploratory tokenisation handoff.

## Verified on 7 September 2026

- Remote MULTI headers contain UT `(8811,21,800,2)` and FT `(8811,21,788,2)`, aligned by sample ID. The adapter uses reference-coordinate mapping, preserves the canonical order and masks twelve absent FT nodes. Header counts alone do not establish final complete-case/split counts.
- Slurm job **25868425**, `dual-MULTI-test-260907`, completed successfully: 64 selected pairs, three epochs, one A100, validation diagnostics. Home staging, scratch execution and archive collection all ran. See `samples/hpc-test-report.md` for evidence and limitations.
- The archive is `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/dual-MULTI-test-260907/`. An isolated home code snapshot was used; the remote Git checkout and ML source data were not edited.
- Nine synthetic contract tests passed, covering encoder counts/calls, masks, joint gradients, affine loss reconstruction, checkpoints, reordered FT mapping, pin calculations, trial configuration and saved artifacts.
- Independent UT/FT field and full-201-point field-to-curve HPO parameter records were recovered. Trial 1 is a transparent capacity/lr compromise, not a dual-HPO optimum.
- Three-epoch smoke predictions remain worse than training-mean baselines. Do not cite the execution test as accuracy, negative-transfer or optimisation evidence.

## Deployment on 10 September 2026

- Full-data Trial 1, Slurm **26267130**, `dual-MULTI-trial1-260910`, completed successfully in 3 h 37 min. Mac download used ignored repo-root `data/`; the user renamed its local folder to `dual-MULTI-trial1`. Windows artifacts now belong in `Z:/p2/`. Remote metadata retains the original dated name.
- Training implementation: Git `7da7e72`. Splits: 7,137 train / 793 validation / 881 locked test; 149 epochs logged, best checkpoint 74. Validation RMSE improvements over the training mean: UT field +5.47%, FT field -23.58%, UT curve +3.42%, FT curve +14.53%. See the dual HPC README for numerical context and limitations; raw weighted-loss sizes alone do not identify the weakest task.
- Nine dual contract tests passed at deployment. Repository contract validation finished with 64 non-failing checks and zero failures. Both saved-run review notebooks executed successfully; generated figures remain in the downloaded smoke run's `results/postProcessing/` directory.

- The completed smoke archive is also downloaded (locally renamed `dual-MULTI-test`); it remains execution evidence, not the default accuracy review. macOS transfer uses `HPC/B3_ML-transfer-mac.sh`; `-windows.sh` preserves the Windows route.
- Generated `data/` and `samples/` content remains ignored. Only `samples/AGENTS.md` is exempted to keep required guidance tracked. Tests no longer depend on ignored sample generators.
- Both Git hosting destinations received the implementation commit, and the clean HPC checkout was fast-forwarded through a verified incremental Git bundle because direct Git SSH authentication was unavailable. Its obsolete origin URL was corrected. No credentials were copied.

## Next evidence

- Notebook corrections verified on 14 September: all 15 dual contract tests passed, including fresh notebook build/train/save/predict, direct HPO dispatch, saved-model reconstruction, sample-ID alignment and coexistence of field viewers. Trial 1 Outputs executed all 14 cells successfully; DualPostProcessing executed all 40 cells with 41 inline plots and no errors. Validation used memory/outside-repository temporary files, not new notebook copies under data.
- At the user's subsequent request, both maintained notebooks were executed on Trial 1 and their outputs saved **in place** under `code/`: 9/9 Outputs and 19/19 PostProcessing code cells, zero errors, 41 inline post-processing plots. Source hashes matched before/after execution, including the user's existing Outputs edit. Training/HPO stayed disabled; raw datasets and archived run artifacts were not changed. These local execution outputs are not additional notebook copies. Interactive widget callbacks require a live kernel and re-execution when reopened.
- Shared HPO notebook executed all 29 cells against an explicitly synthetic Optuna study and saved dual winner: 12 inline plots, no cell errors. The live HPC study was not downloaded or modified. Changed-surface validation passed with 13 non-failing checks; contract validation passed with 67 non-failing checks and zero failures. Abaqus/GPU execution remains outside these local checks.
- The five duplicate notebooks in `data/` were moved to macOS Trash with explicit user approval: `/Users/niccoloforte/.Trash/p2-duplicate-notebooks-20260912/`. Trial 1 copies were full executed versions differing by field selection; smoke copies were older shorter versions. Results, checkpoints and images were preserved. Root instructions now explicitly prohibit surprise scope expansion or executable/notebook copies in data; maintained notebooks stay in p2 code.
- Full-data dual HPO **26375307**, `dual-joint-hpo1`, was submitted on 10 September 2026 through B1 and started at 16:52:57 BST on `sbg9`. Slurm confirms `andrena` / `pilot_andrena`, one A100 40 GB GPU, 12 CPUs, 90,000 MB RAM and **240 hours**. The Python budget is 230 hours, targeting 200 completed/pruned evaluations with up to 450 epochs each. Submission checkout: `8061a43`; `resources/MLdualHPO.py` owns fixed balanced validation ranking, four positive task weights, periodic backups and study-level resume.
- Submit directory: `/data/home/exy053/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1`; log: `dual-joint-hpo1.o26375307`. B1 staged to `/gpfs/scratch/exy053/26375307` and launched Python successfully; study backups/results go to `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1`. Inspect this job before submitting a duplicate or attempting resume. Last verified scheduler state on 10 September was RUNNING; current state and completed training epochs/trials have not been checked in this notebook task.
- Review the labelled pin/interface examples against the intended production geometry. The current profile uses the A1 Ti/Al proportions and the sample's initial disordered coordinates.
- Representative connectivity validation now agrees with one original paired INP specimen; broader agreement remains untested. Do not process thousands of INPs by default.
- The HPO job completed within its budget; reaching the original 200-evaluation target would require an explicitly authorised study resume. Repeated-seed finalist confirmation, locked-test evaluation, full-dimensional optimisation and new FEA verification remain outstanding. Study resume does not continue a partially trained optimizer; interrupted configurations restart from epoch one.
- Local checks cover the expanded architecture, strict reload of the actual Trial 1 checkpoint, HPO archive-to-new-scratch resume, exclusive-lock refusal and saved diagnostic adapters. Repository contract validation previously passed (67 non-failing, zero failures). Remote HPO CLI help, wrapper Bash syntax and Slurm `--test-only` resource validation passed before submission; these do not establish GPU training success.
- SSH access was renewed and the clean HPC checkout fast-forwarded through a verified incremental Git bundle to `8061a43` before submission. Both hosting destinations already contained that implementation. Temporary deployment bundles are outside the repository, not under `data/`.
- Keep the field intermediary and full 201-point curves. User-reported true-field FT full-curve performance is promising; numerical manuscript claims still require the saved evidence.
- SSH access uses the user's active ControlMaster socket. Access availability is temporary and does not itself authorize arbitrary remote changes. Never store passwords in chat or repo files.
- The clean HPC checkout was verified synchronized through `cc06f53` on 17 September, including the notebook corrections. No running HPO's staged source or study was changed.

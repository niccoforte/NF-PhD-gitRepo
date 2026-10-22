# Damage study and dataset tools

`review_damage.py` reviews bounded STATUS products; `review_damage_dataset.py`
reviews the delivered corpus. Their tests remain beside them. The dataset is
complete; none of these commands needs rerunning for ML integration.
`build_damage_banks.py` is the explicit staging CLI for the shared
`resources.damage_data.build_damage_banks` function. `resources/damage.py` maps
original INP elements to parent struts and reduces STATUS with unknown masks;
`damage_data.py` aligns additive allD banks with existing field identities.
These reusable modules serve p1 and p2; they are not competing ML models.

## Agreed next experiment (not implemented/submitted)

Independent UT/FT Case1(i) field generators, width384/eight heads/three layers,
unchanged geometry/context, saved split and masked field MSE. First seed42,
max450 epochs/patience52, final generators only: field-only matched control,
damage-only, field+damage coefficient0.1, field+damage coefficient1. Use the
same normalized masked class-balanced BCE in both auxiliary variants; weak vs
strong changes the coefficient, not the labels or fracture severity. Log both
raw and weighted terms. Fit balancing on training labels only, with bounded
weights; exact balancing convention must be recorded before fitting.

Add a symmetric endpoint/initial-edge-geometry MLP head on native struts, not
new attention tokens. Join by exact specimen ID and initial geometry; drop only
frame0 from21 stored frames, preserving20-frame masks and short UT histories.
Joint runs select checkpoints by the unchanged field criterion; damage-only
selects by an explicitly declared validation damage criterion. Report precision/
recall, PR-AUC, missed/false fractures, onset/path and broken-to-intact reversals,
plus fixed local/global displacement errors. Do not use ordinary binary accuracy
as the primary success metric or silently apply cumulative maximum.

Predicted damage **is an output at inference**. True future damage is never an
input or an instruction to delete connections. First-stage damage feedback is
not part of this experiment. Keep the current loss study untouched.
Confirm useful candidates across42/43/44, then construct training-only OOF
fields and fit unchanged Predicted+True curves. A later, separate curve-input
comparison may test predicted displacement+predicted damage; true-field/true-
damage fits are oracle diagnostics, not deployable results. Do not pay for
OOF/curve retraining before the field-only screen shows a useful result.

## Existing report contracts

- `review_damage_dataset.py` reviews explicit STATUS-only FIELDd products and optional supplied corpus evidence for the FCC dataset report. Selected figures do not establish whole-corpus completion; preserve task/sample identities, unknown masks and complete binary legends. Verify spacing with the full eight-panel suite. It opens no ODB, makes no HPC/archive queries and trains no model. `test_damage_dataset_review.py` checks completion, identity, membership and checksum gates synthetically; run `python -m unittest discover -s p2-DisorderML/HPC/studies/Damage -p test_damage_dataset_review.py -v` from the root.

The `corpus_summary` docstring specifies optional `--execution-json` / `corpus_evidence` schema `fcc_status_corpus_review_v1`: pinned manifest/code hashes, attempts, ordered existing `required_ids`, exact `excluded_ids`, verified transfer rows and four bank proofs. Its producer must reconcile actual files first. Missing attempts become pending in `corpus-attempts.csv`; terminal exceptions retain reasons. Raw frame counts must be positive, not necessarily21: preserve short histories and their last observed endpoints; summary JSON/prose expose frame-count histograms, short-source IDs and required-source counts. Bank axes remain21; reconcile absent sample-frame and observed/total unknown-state counts against each bank's actual source prefixes, and display padding explicitly. `skipped_valid` may reuse selected validation only with matching identity, filename, checksum and code. Never treat excluded-product presence as validation or bank success as raw-corpus completion.

`corpus_complete=true` alone is ignored. Completion requires18,002 explicit unique terminal attempts, existing UT8,882/FT8,928/MULTI8,811 membership with periodic0, no unavailable required members, receipts for every successful product (including existing exclusions), and four verified banks. Bank IDs/order must equal allFIELD, local/HPC NPZ+JSON hashes must match, original/local layout member hashes must agree, and ordered source hashes must join receipts. Use `read_field_layout` hashes (normalized IDs plus shape/dtype/bytes), not ZIP-member hashes. Partial evidence stays incomplete. Corpus totals use batch validation; native ODB comparison remains limited to eight preflight products. `corpus-coverage.png` plots task-specific fracture percentages; JSON/CSV retain evidence and exceptions. Verify the rendered corpus figure alongside the selected eight-panel plots.

- `review_damage.py` validates explicit supported FCC UT/FT `strut_damage_v1` NPZs and builds a conclusions-first visual report without ODB access or training. Plot binary fracture from `strut_failed`/`strut_failure_known`: purple0 for known all-active, red1 for any verified failed member, black dashed for unknown. Do not colour active surface damage grey or use optional scalar severity as the fracture target. Raw SDEG is separate diagnostic context; track whether observed active surface damage later fractures within the saved window, never extrapolate beyond it. Use --ut/--ft for the periodic pair or repeated --case LABEL TASK NPZ for a labelled suite. Keep cases/IDs explicit, retain actual geometry and select FT zooms near the saved nominal tip. Report duplicate inputs/responses without treating filenames as independent designs. Audit-canonical endpoints are not saved ML node order; keep these products separate from displacement loaders.
